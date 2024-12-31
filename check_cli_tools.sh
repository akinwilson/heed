#!/bin/bash 

GREEN='\033[0;32m'
RESET='\033[0m'

if [ -z "$(command nvidia-ctk --version)" ]; then
    echo "Unable to find nvidia container toolkit"
    echo "To install the nvidia container toolkit, please follow this guide: ${GREEN}https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html${RESET}"
    exit 1 
fi 



echo "You have all the required CLI tools to begin."
