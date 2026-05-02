BOARDS += amd_npu

local-amd_npu: version
	docker buildx bake --file=docker/amd_npu/amd_npu.hcl amd_npu \
		--set amd_npu.tags=frigate:latest-amd-npu \
		--load

build-amd_npu: version
	docker buildx bake --file=docker/amd_npu/amd_npu.hcl amd_npu \
		--set amd_npu.tags=$(IMAGE_REPO):${GITHUB_REF_NAME}-$(COMMIT_HASH)-amd-npu

push-amd_npu: build-amd_npu
	docker buildx bake --file=docker/amd_npu/amd_npu.hcl amd_npu \
		--set amd_npu.tags=$(IMAGE_REPO):${GITHUB_REF_NAME}-$(COMMIT_HASH)-amd-npu \
		--push
