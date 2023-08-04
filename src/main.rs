// a simple server implementation for smaragdine on linux. only one process can be monitored at a time.
// /proc/stat, /proc/pid/task, and /sys/class/powercap are each sampled from their own thread.
mod protos {
    tonic::include_proto!("smaragdine.protos.sample");
}

use std::fs;
use std::io;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use std::vec::Vec;

use clap::App;

use env_logger;

use log;
use log::{LevelFilter, error, info, warn};

use procfs::{CpuInfo, CpuTime, KernelStats, ProcError};
use procfs::process::{Process, Stat};

use nvml_wrapper::NVML;

use tonic::{Request, Response, Status};
use tonic::transport::Server;

use protos::Sample;
use protos::{CpuSample, CpuReading};
use protos::{NvmlSample, NvmlReading};
use protos::{RaplSample, RaplReading};
use protos::{ProcessSample, TaskReading};
use protos::{DataSet, StartRequest, StartResponse, StopRequest, StopResponse, ReadRequest, ReadResponse, sample::Data};
use protos::sampler_server::{Sampler, SamplerServer};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

enum SamplingError {
    NotRetryable(String),
    //  - rapl doesn't exist
    //  - the process dies
    RequiresTermination(String),
    Retryable(String),
}

// code to sample /proc/stat
fn sample_cpus() -> Result<Sample, SamplingError> {
    match read_cpus() {
        Ok(stats) => {
            let mut sample = CpuSample::default();
            sample.timestamp = now_ms();
            stats.into_iter().for_each(|stat| sample.reading.push(stat));
            let mut s = Sample::default();
            s.data = Some(Data::Cpu(sample));
            Ok(s)
        },
        Err(ProcError::PermissionDenied(e)) | Err(ProcError::NotFound(e)) =>
            Err(SamplingError::NotRetryable(format!("/proc/stat could not be read: {:?}", e))),
        _ => Err(SamplingError::Retryable("unable to read cpus".to_string()))
    }
}

fn read_cpus() -> Result<Vec<CpuReading>, ProcError> {
    Ok(KernelStats::new()?.cpu_time
        .into_iter()
        .enumerate()
        .map(|(cpu, stat)| cpu_stat_to_proto(cpu as u32, stat))
        .collect())
}

fn cpu_stat_to_proto(cpu: u32, stat: CpuTime) -> CpuReading {
    let mut stat_proto = CpuReading::default();
    stat_proto.cpu = cpu;
    stat_proto.socket = CpuInfo::new().unwrap().physical_id(cpu as usize).unwrap();
    stat_proto.user = Some(stat.user as u32);
    stat_proto.nice = Some(stat.nice as u32);
    stat_proto.system = Some(stat.system as u32);
    stat_proto.idle = Some(stat.idle as u32);
    if let Some(jiffies) = stat.iowait {
        stat_proto.iowait = Some(jiffies as u32);
    };
    if let Some(jiffies) = stat.irq {
        stat_proto.irq = Some(jiffies as u32);
    };
    if let Some(jiffies) = stat.softirq {
        stat_proto.softirq = Some(jiffies as u32);
    };
    if let Some(jiffies) = stat.steal {
        stat_proto.steal = Some(jiffies as u32);
    };
    if let Some(jiffies) = stat.guest {
        stat_proto.guest = Some(jiffies as u32);
    };
    if let Some(jiffies) = stat.guest_nice {
        stat_proto.guest_nice = Some(jiffies as u32);
    };
    stat_proto
}

// code to sample /proc/[pid]/task/[tid]/stat
fn sample_tasks(pid: i32) -> Result<Sample, SamplingError> {
    match read_tasks(pid) {
        Ok(tasks) => {
            let mut sample = ProcessSample::default();
            sample.timestamp = now_ms();
            tasks.into_iter().for_each(|s| sample.reading.push(s));
            let mut s = Sample::default();
            s.data = Some(Data::Process(sample));
            Ok(s)
        }
        Err(ProcError::PermissionDenied(_)) | Err(ProcError::NotFound(_)) => Err(SamplingError::RequiresTermination(format!("/proc/{}/task could not be read", pid))),
        _ => Err(SamplingError::Retryable("unable to read tasks".to_string()))
    }
}

fn read_tasks(pid: i32) -> Result<Vec<TaskReading>, ProcError> {
    Ok(Process::new(pid)?.tasks()?
        .flatten()
        .filter_map(|stat| stat.stat().ok())
        .map(|stat| task_stat_to_proto(stat))
        .collect())
}

fn task_stat_to_proto(stat: Stat) -> TaskReading {
    let mut stat_proto = TaskReading::default();
    stat_proto.task_id = stat.pid as u32;
    if let Some(cpu) = stat.processor {
        stat_proto.cpu = cpu as u32;
        // this can bloat the data
        // stat_proto.name = stat.comm;
        stat_proto.user = Some(stat.utime as u32);
        stat_proto.system = Some(stat.stime as u32);
    };
    stat_proto
}

// code to sample /sys/class/powercap
fn sample_rapl() -> Result<Sample, SamplingError> {
    let readings = read_rapl()?;
    let mut sample = RaplSample::default();
    sample.timestamp = now_ms();
    readings.into_iter().for_each(|reading| sample.reading.push(reading));
    let mut s = Sample::default();
    s.data = Some(Data::Rapl(sample));
    Ok(s)
}

static POWERCAP_PATH: &str = "/sys/class/powercap";
// TODO: we need a way to report if rapl isn't readable (no sudo for example); the current way missed it
fn read_rapl() -> Result<Vec<RaplReading>, SamplingError> {
    match fs::read_dir(POWERCAP_PATH) {
        Ok(components) => Ok(components
            .filter_map(|e| e.unwrap().file_name().into_string().ok())
            .filter(|f| f.contains("intel-rapl") && f.matches(":").count() == 1)
            .map(|f| read_socket(f.split(":").nth(1).unwrap().parse().unwrap()))
            .filter_map(Result::ok)
            .collect()),
        Err(e) => match e.kind() {
            io::ErrorKind::PermissionDenied | io::ErrorKind::NotFound =>
                Err(SamplingError::NotRetryable(format!("{} could not be read: {:?}", POWERCAP_PATH.to_string(), e))),
            _ => Err(SamplingError::Retryable("unable to read rapl".to_string())),
        }
    }
}

fn read_socket(socket: u32) -> Result<RaplReading, io::Error> {
    let mut reading = RaplReading::default();
    reading.socket = socket;
    reading.package = Some(parse_rapl_energy(format!("{}/intel-rapl:{}/energy_uj", POWERCAP_PATH, socket))?);
    reading.dram = Some(parse_rapl_energy(format!("{}/intel-rapl:{}:0/energy_uj", POWERCAP_PATH, socket))?);
    Ok(reading)
}

fn parse_rapl_energy(rapl_energy_file: String) -> Result<u64, io::Error> {
    Ok(fs::read_to_string(rapl_energy_file)?.trim().parse().unwrap())
}

fn sample_nvml(nvml: Arc<NVML>) -> Result<Sample, SamplingError> {
    match nvml.device_count() {
        Ok(device_count) => {
            // TODO(timur): how many devices do we examine?
            let mut sample = NvmlSample::default();
            sample.timestamp = now_ms();
            for idx in 0..device_count {
                let device = nvml.device_by_index(idx).unwrap();
                let mut reading = NvmlReading::default();
                reading.index = idx as u32;
                reading.bus_id = device.pci_info().unwrap().bus_id;
                reading.power_usage = Some(device.power_usage().unwrap());
                sample.reading.push(reading);
            }
            let mut s = Sample::default();
            s.data = Some(Data::Nvml(sample));
            Ok(s)
        },
        // TODO(timur): there are a handful of cases we should catch. how do we break this up?
        Err(e) => Err(SamplingError::NotRetryable(format!("{:?}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use procfs::CpuInfo;
    use procfs::process::Process;

    use crate::protos::sample::Data;

    #[test]
    // make sure sample_cpus returns the right data type
    fn jiffies_smoke_test() {
        let start = now_ms();
        if let Ok(sample) = sample_cpus() {
            if let Some(Data::Cpu(sample)) = sample.data {
                assert!(sample.timestamp <= now_ms());
                assert!(sample.timestamp >= start);
                assert_eq!(sample.reading.len(), CpuInfo::new().unwrap().num_cores());
            } else {
                panic!("sampling cpus failed; data other than CpuSample returned");
            };
        } else {
            panic!("sampling cpus failed; /proc/stat couldn't be read");
        };

        let start = now_ms();
        let me = Process::myself().unwrap();
        if let Ok(sample) = sample_tasks(me.pid) {
            if let Some(Data::Process(sample)) = sample.data {
                assert!(sample.timestamp <= now_ms());
                assert!(sample.timestamp >= start);
                assert_eq!(sample.reading.len(), me.tasks().unwrap().count());
            } else {
                panic!("sampling tasks failed; data other than ProcessSample returned");
            };
        } else {
            panic!("sampling tasks failed; /proc/[pid]/task couldn't be read");
        };
    }

    #[test]
    // make sure we have rapl (/sys/class/powercap) and that we read all the values
    fn rapl_smoke_test() {
        let start = now_ms();
        if let Ok(sample) = sample_rapl() {
            if let Some(Data::Rapl(sample)) = sample.data {
                assert!(sample.timestamp <= now_ms());
                assert!(sample.timestamp >= start);
            } else {
                panic!("sampling rapl failed; data other than RaplSample returned");
            };
        } else {
            panic!("sampling rapl failed; /sys/class/powercap couldn't be read");
        };
    }

    #[test]
    // make sure we have nvml (libnvidia-ml.so) and that we read all the values
    fn nvml_smoke_test() {
        let start = now_ms();
        if let Ok(nvml) = NVML::init() {
            if let Ok(sample) = sample_nvml() {
                if let Some(Data::Nvml(sample)) = sample.data {
                    assert!(sample.timestamp <= now_ms());
                    assert!(sample.timestamp >= start);
                } else {
                    panic!("sampling nvml failed; data other than NvmlSample returned");
                };
            } else {
                panic!("sampling nvml failed; libnvidia-ml.so could not be found");
            };
        } else {
            panic!("sampling nvml failed; libnvidia-ml.so could not be found");
        };
    }
}

// sampler implementation
struct SamplerImpl {
    nvml: Arc<NVML>,
    period: Duration,
    is_running: Arc<AtomicBool>,
    sender: Arc<Mutex<Sender<Sample>>>,
    receiver: Arc<Mutex<Receiver<Sample>>>,
}

impl SamplerImpl {
    fn start_sampling_from<F>(&self, mut source: F, period: Duration)
        where F: FnMut() -> Result<Sample, SamplingError> + Send + Sync + Clone + 'static {
        let is_running = self.is_running.clone();
        let sender = self.sender.clone();
        thread::spawn(move || {
                while is_running.load(Ordering::Relaxed) {
                    let start = Instant::now();
                    match source() {
                        Ok(sample) => sender.lock().unwrap().send(sample).unwrap(),
                        Err(SamplingError::NotRetryable(message)) => {
                            error!("there was an error with a source that cannot be retried: {}", message);
                            break;
                        },
                        Err(SamplingError::RequiresTermination(message)) => {
                            error!("there was an error that requires smaragdine to stop sampling: {}", message);
                            is_running.store(false, Ordering::Relaxed);
                            break;
                        }
                        Err(SamplingError::Retryable(message)) => error!("there was an error with a source that will be retried: {}", message),
                    };

                    let now = Instant::now() - start;
                    if period > now { thread::sleep(period - now); }
                }
        });
    }
}

impl SamplerImpl {
    fn new(nvml: NVML, period_millis: u64) -> SamplerImpl {
        let (tx, rx) = channel::<Sample>();
        SamplerImpl {
            nvml: Arc::new(nvml),
            period: Duration::from_millis(period_millis),
            is_running: Arc::new(AtomicBool::new(false)),
            sender: Arc::new(Mutex::new(tx)),
            receiver: Arc::new(Mutex::new(rx)),
        }
    }
}

#[tonic::async_trait]
impl Sampler for SamplerImpl {
    async fn start(
        &self,
        request: Request<StartRequest>,
    ) -> Result<Response<StartResponse>, Status> {
        if !self.is_running.load(Ordering::Relaxed) {
            // check the pid first in case we need to abandon?
            let r = request.into_inner();
            let pid: i32 = r.pid.unwrap() as i32;
            let period: Duration = Duration::from_millis(
                r.period.unwrap_or(self.period.as_millis() as u64));
            info!("start requested for pid={} at {:?}", pid, period);

            self.is_running.store(true, Ordering::Relaxed);

            let pid1 = pid.clone();
            self.start_sampling_from(sample_cpus, period);
            self.start_sampling_from(move || sample_tasks(pid1), period);

            self.start_sampling_from(sample_rapl, period);

            let nvml = self.nvml.clone();
            self.start_sampling_from(move || sample_nvml(nvml.clone()), period);
        } else {
            warn!("ignoring start request while collecting");
        }

        Ok(Response::new(StartResponse {}))
    }

    async fn stop(
        &self,
        _: Request<StopRequest>,
    ) -> Result<Response<StopResponse>, Status> {
        info!("stop requested");
        self.is_running.store(false, Ordering::Relaxed);
        Ok(Response::new(StopResponse {}))
    }

    async fn read(
        &self,
        _: Request<ReadRequest>,
    ) -> Result<Response<ReadResponse>, Status> {
        info!("read requested");
        let response = if !self.is_running.load(Ordering::Relaxed) {
            let mut data = DataSet::default();
            let receiver = self.receiver.lock().unwrap();
            while let Ok(sample) = receiver.try_recv() {
                match sample.data {
                    Some(Data::Cpu(sample)) => data.cpu.push(sample),
                    Some(Data::Nvml(sample)) => data.nvml.push(sample),
                    Some(Data::Rapl(sample)) => data.rapl.push(sample),
                    Some(Data::Process(sample)) => data.process.push(sample),
                    _ => log::warn!("no sample found!")
                }
            }
            ReadResponse {data: Some(data)}
        } else {
            warn!("ignoring read request while collecting");
            ReadResponse {data: None}
        };
        Ok(Response::new(response))
    }
}

#[tokio::main]
async fn main() -> Result<(), Arc<dyn std::error::Error>> {
    env_logger::builder().filter_level(LevelFilter::Info).init();
    let matches = App::new("smaragdine")
        .arg_from_usage("--addr [address] 'The address to host the smaragdine server at'")
        .arg_from_usage("--period [period] 'The default period in milliseconds to collect at'")
        .get_matches();

    let addr = matches.value_of("addr").or(Some("[::1]:50051")).unwrap().parse().unwrap();
    info!("smaragdine listening on {}", addr);

    // should only load this once and use the reference:
    // https://docs.rs/nvml-wrapper/latest/nvml_wrapper/struct.Nvml.html#method.init
    if let Ok(nvml) = NVML::init() {
        let sampler = match matches.value_of("period") {
            Some(period) => SamplerImpl::new(nvml, period.parse().unwrap()),
            _ => SamplerImpl::new(nvml, 4),
        };
        Server::builder()
            .add_service(SamplerServer::new(sampler))
            .serve(addr)
            .await
            .unwrap();
    }
    Ok(())
}
