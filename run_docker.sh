# http://fabiorehm.com/blog/2014/09/11/running-gui-apps-with-docker/
# https://stackoverflow.com/a/28971413
# assumes that image was built with command:
# docker build -t robosuite .

DOCKER_NAME=${USER}-robosuite

# stop and remove any already running containers
EXISTING_DOCKER_CONTAINER_ID=`docker ps -aq -f name=${DOCKER_NAME}`
if [ ! -z "${EXISTING_DOCKER_CONTAINER_ID}" ]; then
	docker stop ${EXISTING_DOCKER_CONTAINER_ID}
	docker rm ${EXISTING_DOCKER_CONTAINER_ID}
fi

# sharing git authentication to docker containiner:
# https://embeddeduse.com/2023/03/20/accessing-private-git-repositories-from-docker-containers/

docker run \
	-it \
	-d \
	--gpus all \
	-e DISPLAY=$DISPLAY \
	-e "USER_ID=$(id -u)" \
	-e GROUP_ID="$(id -g)" \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v "${HOME}/.ssh:/home/builder/.ssh:rw" \
	-v "${SSH_AUTH_SOCK}:/ssh.socket" -e "SSH_AUTH_SOCK=/ssh.socket" \
	--mount type=bind,source=$(pwd),target=/srv \
	--shm-size=1024m \
	--name ${DOCKER_NAME} \
	robosuite \
	bash
