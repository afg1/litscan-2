fn main() {
    // CI passes LITSCAN_GIT_SHA in explicitly: git inside the manylinux container
    // refuses the bind-mounted repo (dubious ownership), and sdists have no .git.
    let sha = std::env::var("LITSCAN_GIT_SHA").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| {
        std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".into())
    });
    println!("cargo:rustc-env=LITSCAN_GIT_SHA={sha}");
    println!("cargo:rerun-if-env-changed=LITSCAN_GIT_SHA");
    println!("cargo:rerun-if-changed=.git/HEAD");
    // HEAD only changes on checkout; a commit moves the branch ref instead.
    println!("cargo:rerun-if-changed=.git/refs/heads");
}
