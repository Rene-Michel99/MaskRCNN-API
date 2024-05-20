# Instance Segmentation Task using Mask RCNN

This repository contains a Flask server that provides the instance segmentation task using the Mask RCNN architecture. Additionally, a Swagger was implemented to provide details on how to use an API.

## How it works

Mask RCNN is a convolutional neural network architecture that performs instance detection and segmentation in images. It is capable of identifying different objects in an image and segmenting them individually, assigning a mask to each detected object.

This repository uses a pre-trained implementation of Mask RCNN to perform instance segmentation on images provided via a RESTful API.

## How to use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   cd nome-do-repositorio
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Mask RCNN:**
   ```bash
   git clone -b develop https://github.com/Rene-Michel99/Mask-RCNN-TF2.8
   pip install ./Mask-RCNN-TF2.8
   ```

3. **Start the server:**
   ```bash
   python app.py
   ```

   The Flask server will start and be ready to receive requests.

4. **Access Swagger:**
   Open the `swagger.yaml` to access the API documentation via Swagger. There you will find details about the available endpoints and how to use each one.

5. **Send Requests:**
   You can send HTTP requests to the Flask server using any HTTP client or library you prefer. See the Swagger documentation for details on required parameters and expected response formats.

## Using Docker

This project also includes a Dockerfile to facilitate containerized deployment.

1. **Build the Docker Image:**
   ```bash
   docker build -t nome-da-imagem .
   ```

2. **Execute o Contêiner Docker:**
   ```bash
   docker run -p 5000:5000 nome-da-imagem
   ```

   The Flask server will be started inside the container and will be accessible from `http://localhost:5000`.

## Contribution

Contributions are welcome! If you find bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).