# Human Pose Estimation Model Serving on Multi-Model-Server (MMS)

## Installation

1. Build the image for the MMS:
    ```bash
    $ docker build -t human-pose-mms .
    ```
2. Prepare the deployment model with model-archiver:
    ```bash
    $ docker run --rm -it -v $(pwd):/tmp human-pose-mms:latest bash
    
    $ model-archiver --model-name pose --model-path /tmp/service --handler handler:handle --runtime python3 --export-path /tmp
    ```
3. Start the MMS:
    ```bash
    $ docker run --rm -it -p 8080:8080 -p 8081:8081 -v $(pwd):/tmp mms-human-pose:latest multi-model-server --start --mms-config /tmp/config.properties --models posenet=pose.mar --model-store /tmp
    ```

##  Testing

Send a request with curl:
```bash
curl -X POST http://localhost:8080/predictions/posenet/ -T example.jpg
```

Thanks to the [Joshua Earle](https://unsplash.com/@joshuaearle) for [photo on Unsplash](https://unsplash.com/photos/ICE__bo2Vws)
