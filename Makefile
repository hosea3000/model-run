.PHONY: init

buildx:
	docker buildx build -f Dockerfile --platform linux/amd64 -t hkccr.ccs.tencentyun.com/sijiu-test/image-embedding-api:latest . --push

docker_login:
	echo "He4112043." | docker login hkccr.ccs.tencentyun.com --username=100042344709 --password-stdin
