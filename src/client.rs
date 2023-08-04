// a simple client that talks to an smaragdine server. currently assumes the server can only watch
// a single process
mod protos {
    tonic::include_proto!("smaragdine.protos.sample");
}

use std::fs::File;
use std::io::Write;

use clap::App;
use prost::Message;

use protos::sampler_client::SamplerClient;
use protos::{ReadRequest, StartRequest, StopRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("smaragdine")
        .subcommand(App::new("start").about("begins sampling")
            .arg_from_usage("--pid=<pid> 'The id of the process to monitor'")
            .arg_from_usage("--period=<period> 'The sampling period'"))
        .subcommand(App::new("stop").about("stops sampling"))
        .subcommand(App::new("read").about("reads most recent samples from the server")
            .arg_from_usage("--output=[output] 'The path to write the data set to'"))
        .subcommand(App::new("smoke_test").about("pings the server to make sure it runs successfully and produces data"))
        .arg_from_usage("--addr [address] 'The address the smaragdine server is hosted at'")
        .get_matches();
    let (cmd, submatches) = matches.subcommand();

    let mut client = SamplerClient::connect("http://[::1]:50051").await?;
    match cmd {
        "start" => {
            client
                .start(tonic::Request::new(StartRequest {
                    pid: submatches.unwrap().value_of("pid").unwrap().parse().ok(),
                    period: submatches.unwrap().value_of("period").unwrap().parse().ok()
                }))
                .await?;
        },
        "stop" => {
            client.stop(tonic::Request::new(StopRequest { pid: None })).await?;
        }
        "read" => {
            let message = client
                .read(tonic::Request::new(ReadRequest { pid: None }))
                .await?;
            let message = message.get_ref().data.as_ref().unwrap();
            match submatches.unwrap().value_of("output") {
                Some(path) => {
                    let mut buffer = vec![];
                    match message.encode(&mut buffer) {
                        Ok(_) => {
                            let mut file = File::create(path)?;
                            file.write_all(&buffer)?;
                        }
                        Err(e) => println!("error encoding message: {}", e)
                    }
                }
                _ => println!("{:?}", message)
            }
        }
        "smoke_test" => {
            client
                .start(tonic::Request::new(StartRequest { pid: Some(1), period: None }))
                .await?;
            std::thread::sleep(std::time::Duration::from_secs(1));
            client
                .stop(tonic::Request::new(StopRequest { pid: None }))
                .await?;
            let message = client
                .read(tonic::Request::new(ReadRequest { pid: None }))
                .await?;
            let message = message.get_ref().data.as_ref().unwrap();
            let mut empty = Vec::new();
            if message.cpu.len() == 0 {
                empty.push("cpu");
            }
            if message.process.len() == 0 {
                empty.push("process");
            }
            if message.rapl.len() == 0 {
                empty.push("rapl");
            }
            if message.nvml.len() == 0 {
                empty.push("nvml");
            }
            if empty.len() > 0 {
                println!("{:?} had no data", empty);
            }
        }
        _ => println!("don't understand {}", cmd),
    };

    Ok(())
}
