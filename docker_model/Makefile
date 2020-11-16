IMAGE_TAG := "txpptf"

build:
	docker build -t ${IMAGE_TAG} -f Dockerfile ../

run:
	# Make sure directories exist to prevent permission issues
	docker run \
		--rm \
		-v `pwd`/..:/workspace \
		${IMAGE_TAG}

interactive:
	# Make sure directories exist to prevent permission issues
	docker run \
		-it \
		--rm \
		-v `pwd`/..:/workspace \
		${IMAGE_TAG} \
 		/bin/bash
