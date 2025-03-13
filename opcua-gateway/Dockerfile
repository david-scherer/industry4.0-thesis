# Use the official Go image as the base
FROM golang:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the Go module files (go.mod and go.sum)
COPY go.mod go.sum ./

# Download the Go modules
RUN go mod download

# Copy the rest of your application code
COPY . .
COPY ./internal/config/config-docker.yml /app/internal/config/config.yml

# Build the Go application (assuming your main file is in the 'cmd' directory)
RUN go build -o ./cmd/opcua-gateway ./cmd

EXPOSE 8880

# Run the application when the container starts
CMD ["./cmd/opcua-gateway"]