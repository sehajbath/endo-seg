#!/bin/bash

# Script to download UT-EndoMRI dataset from Zenodo
# Dataset: https://zenodo.org/records/15750762
# Size: ~8.0 GB

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ZENODO_RECORD="15750762"
DATASET_URL="https://zenodo.org/records/${ZENODO_RECORD}/files/UT-EndoMRI.zip"
DATA_DIR="data/raw"
DOWNLOAD_PATH="${DATA_DIR}/UT-EndoMRI.zip"
EXTRACT_DIR="${DATA_DIR}/UT-EndoMRI"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}UT-EndoMRI Dataset Download Script${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -c"
    echo -e "${GREEN}Using wget for download${NC}"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -C - -o"
    echo -e "${GREEN}Using curl for download${NC}"
else
    echo -e "${RED}Error: Neither wget nor curl is available${NC}"
    echo "Please install wget or curl and try again"
    exit 1
fi

# Create data directory
echo -e "\n${YELLOW}Creating data directory...${NC}"
mkdir -p ${DATA_DIR}

# Check if dataset already exists
if [ -d "${EXTRACT_DIR}" ]; then
    echo -e "${YELLOW}Dataset directory already exists: ${EXTRACT_DIR}${NC}"
    read -p "Do you want to re-download and extract? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Skipping download. Existing dataset will be used.${NC}"
        exit 0
    fi
fi

# Download dataset
if [ ! -f "${DOWNLOAD_PATH}" ]; then
    echo -e "\n${YELLOW}Downloading UT-EndoMRI dataset (~8.0 GB)...${NC}"
    echo "Source: ${DATASET_URL}"

    if [[ $DOWNLOAD_CMD == "wget -c" ]]; then
        wget -c ${DATASET_URL} -O ${DOWNLOAD_PATH}
    else
        curl -L -C - -o ${DOWNLOAD_PATH} ${DATASET_URL}
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Download completed successfully!${NC}"
    else
        echo -e "${RED}Download failed!${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Dataset archive already exists: ${DOWNLOAD_PATH}${NC}"
    read -p "Do you want to re-download? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm ${DOWNLOAD_PATH}
        if [[ $DOWNLOAD_CMD == "wget -c" ]]; then
            wget -c ${DATASET_URL} -O ${DOWNLOAD_PATH}
        else
            curl -L -C - -o ${DOWNLOAD_PATH} ${DATASET_URL}
        fi
    fi
fi

# Verify download
echo -e "\n${YELLOW}Verifying download...${NC}"
if [ ! -f "${DOWNLOAD_PATH}" ]; then
    echo -e "${RED}Error: Downloaded file not found!${NC}"
    exit 1
fi

FILE_SIZE=$(stat -f%z "${DOWNLOAD_PATH}" 2>/dev/null || stat -c%s "${DOWNLOAD_PATH}" 2>/dev/null)
echo "Downloaded file size: $(numfmt --to=iec-i --suffix=B ${FILE_SIZE} 2>/dev/null || echo ${FILE_SIZE} bytes)"

# Extract dataset
echo -e "\n${YELLOW}Extracting dataset...${NC}"
if command -v unzip &> /dev/null; then
    unzip -q ${DOWNLOAD_PATH} -d ${DATA_DIR}
    echo -e "${GREEN}Extraction completed!${NC}"
else
    echo -e "${RED}Error: unzip command not found${NC}"
    echo "Please install unzip and manually extract ${DOWNLOAD_PATH}"
    exit 1
fi

# Verify extraction
echo -e "\n${YELLOW}Verifying extraction...${NC}"
if [ -d "${EXTRACT_DIR}/D1_MHS" ] && [ -d "${EXTRACT_DIR}/D2_TCPW" ]; then
    echo -e "${GREEN}Dataset extracted successfully!${NC}"

    # Count subjects
    D1_COUNT=$(ls -d ${EXTRACT_DIR}/D1_MHS/D1-* 2>/dev/null | wc -l)
    D2_COUNT=$(ls -d ${EXTRACT_DIR}/D2_TCPW/D2-* 2>/dev/null | wc -l)

    echo -e "\nDataset contents:"
    echo "  Dataset 1 (D1_MHS): ${D1_COUNT} subjects"
    echo "  Dataset 2 (D2_TCPW): ${D2_COUNT} subjects"

    # Optional: Remove zip file
    read -p "Do you want to remove the downloaded zip file to save space? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm ${DOWNLOAD_PATH}
        echo -e "${GREEN}Zip file removed.${NC}"
    fi
else
    echo -e "${RED}Error: Expected directories not found after extraction${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Download and setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nDataset location: ${EXTRACT_DIR}"
echo -e "\nNext steps:"
echo "  1. Create data splits: python scripts/create_splits.py"
echo "  2. Explore dataset: python scripts/explore_data.py"
echo "  3. Start preprocessing: python scripts/preprocess_data.py"

exit 0