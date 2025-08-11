# hey-kevin-backend

This container hosts the server responsible for handling HTTP requests and responses to perform image segmentation with MobileSAM. It processes incoming images, applies segmentation, and returns the modified images with masks. The output images are stored in local directory app/lib/assets/segment

## Getting Started

### Prerequisites
- In order to run this container you'll need docker installed.
- Windows
- OS X
- Linux

### Usage
1. Clone repo.<br>
    `git clone https://github.com/EegArlert/hey-kevin-backend`
2. Create .env file for this repo<br>
    `cd hey-kevin-backend`<br>
    `// make file (touch .env or whatever)`
3. Build image from dockerfile<br>
    `cd image_processing`<br>
    `docker build -t "hey_kevin_backend_v1" .`<br>
    `cd ..`
4. Find IP Address on local network  <br>
    a. Windows<br>
        `ipconfig /all`<br>
        Find `IPv4 Address` under the `Wireless LAN adapter Wi-Fi:` section. Lets call it `local_ip`.<br>
    b. Apple
5. Add variables to .env<br>
    `DOCKER_IMAGE_NAME=hey_kevin_backend_v1`<br>
    `LOCALHOST_PICTURE_PATH=http://<local_ip>:8000/images/cropped_segmented.jpg`
6. Run container<br>
    `docker compose up`
