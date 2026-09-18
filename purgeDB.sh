#!/bin/sh
# Clears out the Chroma database and restarts the service. Use with caution.
./shutdown.sh
./litsearch.sh purge --yes
./startup.sh
