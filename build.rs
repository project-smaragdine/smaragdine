extern crate tonic_build;

fn main() -> std::io::Result<()> {
    tonic_build::configure()
        .compile(&[
            "protos/jiffies.proto",
            "protos/nvml.proto",
            "protos/rapl.proto",
            "protos/sample.proto",
            "protos/sampler.proto"
        ], &["."])?;
    Ok(())
}
