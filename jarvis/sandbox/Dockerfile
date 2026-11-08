# Steganography CTF sandbox. Contains ONLY the restricted tool set the agent may use
# (config.ALLOWED_PROGRAMS). Runs with no network, as a non-root user.
FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      coreutils findutils grep file binutils xxd bsdextrautils \
      libimage-exiftool-perl binwalk steghide foremost ffmpeg sox \
      ruby \
 && gem install --no-document zsteg \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -u 1000 -m ctf
USER ctf
WORKDIR /work
CMD ["sleep", "infinity"]
