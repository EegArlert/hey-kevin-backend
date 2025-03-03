# hey-kevin-backend

This container hosts the YOLO server responsible for handling HTTP requests and responses to perform image segmentation. It processes incoming images, applies segmentation, and returns the modified images with masks. The output images are stored in local directory app/lib/assets/segment

## Getting Started

### Prerequisites
- In order to run this container you'll need docker installed.
- Windows
- OS X
- Linux

### Usage
1. Clone repo.
    `git clone https://github.com/EegArlert/hey-kevin-backend`
2. Create .env file for this repo
    `cd hey-kevin-backend`
    `// make file (touch .env or whatever)`
3. Build image from dockerfile
    `cd image_processing`
    `docker build -t "hey_kevin_backend_v1" .`
    `cd ..`
4. Find IP Address on local network  
    a. Windows
        `ipconfig /all`
        Find `IPv4 Address` under the `Wireless LAN adapter Wi-Fi:` section. Lets call it `local_ip`.
    b. Apple
5. Add variables to .env
    `DOCKER_IMAGE_NAME=hey_kevin_backend_v1`
    `LOCALHOST_PICTURE_PATH=http://<local_ip>:8000/images/cropped_segmented.jpg`
6. Run container
    `docker compose up`