variable "RYZEN_AI_VERSION" {
  default = "1.3.1"
}

target wget {
  dockerfile = "docker/main/Dockerfile"
  platforms = ["linux/amd64"]
  target = "wget"
}

target deps {
  dockerfile = "docker/main/Dockerfile"
  platforms = ["linux/amd64"]
  target = "deps"
}

target rootfs {
  dockerfile = "docker/main/Dockerfile"
  platforms = ["linux/amd64"]
  target = "rootfs"
}

target amd_npu {
  dockerfile = "docker/amd_npu/Dockerfile"
  contexts = {
    deps = "target:deps",
    wget = "target:wget",
    rootfs = "target:rootfs"
  }
  platforms = ["linux/amd64"]
  args = {
    RYZEN_AI_VERSION = RYZEN_AI_VERSION
  }
}
