# Build stage — workspace build of the server binary only.
FROM rust:1.83-slim AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p vex-server

# Runtime stage — distroless-ish slim image, non-root.
FROM debian:bookworm-slim
RUN useradd --system --uid 1000 vex && mkdir -p /data && chown vex /data
COPY --from=builder /build/target/release/vex-server /usr/local/bin/vex-server
USER vex
ENV VEX_ADDR=0.0.0.0:8080 \
    VEX_DATA_DIR=/data
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["vex-server"]
