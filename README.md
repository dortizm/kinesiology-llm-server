
# Project Kinesiology LLM Server

This project uses `docker-compose` to run three services: `ollama`, `langserve`, and `chroma`. Each service is defined in the `docker-compose.yml` file and is part of a common network called `mynetwork`.

## Prerequisites

- Docker (>= 20.10.0)
- Docker Compose (>= 1.29.0)
- Access to a GPU (for the `ollama` service)

## Services

### ollama
- Image: `ollama/ollama`
- Requires GPU (NVIDIA)
- Exposed port: `11434`

### langserve
- Image: `langserve:latest`
- Built from the `./langserve` context
- Exposed port: `8001`

### chroma
- Image: `ghcr.io/chroma-core/chroma:latest`
- Volume: `index_data:/chroma/.chroma/index`
- Exposed port: `8000`

## Usage Instructions

### Clone the repository

Clone this repository to your local machine:

```bash
git clone https://github.com/dortizm/kinesiology-llm-server
cd kinesiology-llm-server
```

### Build and run the services

To start the services defined in the `docker-compose.yml` file, use the following command:

```bash
docker-compose up --build
```

This command will build the `langserve` image and start the services with the mapped ports on your local machine.

### Access the services

- `ollama` will be available at `http://localhost:11434`
- `langserve` will be available at `http://localhost:8001`
- `chroma` will be available at `http://localhost:8000`

### Stop the services

To stop and remove the containers:

```bash
docker-compose down
```

This command will stop all the services and remove the containers.

## Volumes

- `chroma` uses a local volume called `index_data` to persist data in `/chroma/.chroma/index`.

## Networks

The services are connected to a network called `mynetwork` with a `bridge` driver.

## Additional Notes

- Ensure that NVIDIA drivers are installed and properly configured to use the GPU for the `ollama` service.
