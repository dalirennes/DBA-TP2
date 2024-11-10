

  //nn = new NeuralNetwork(784, 128, 64, 10, "/Users/dada/eit/1. curriculum/3. BDA (DataAnalsis)/MyTP/BDA-TP2/model_weights.json");
import peasy.PeasyCam;
PeasyCam visualizationCam;

JSONObject weightsJSON;
float[][] w1, w2, w3;
float[][] inputMat = new float[1][784];
boolean visualizeLines = false;

int screen_w = 280;

PGraphics canvas, visualization;

void setup() {
  size(1000, 500, P3D);
  canvas = createGraphics(screen_w, screen_w);
  visualization = createGraphics(500, 500, P3D);

  canvas.beginDraw();
  canvas.background(0);
  canvas.endDraw();
  
  visualization.beginDraw();
  visualization.background(0);
  visualization.endDraw();

  visualizationCam = new PeasyCam(this, visualization, 400);

  // Load JSON file containing weights
  weightsJSON = loadJSONObject("/Users/dada/eit/1. curriculum/3. BDA (DataAnalsis)/MyTP/BDA-TP2/model_weights.json");

  // Load weights for each layer
  w1 = loadMatrix(weightsJSON, "w1");
  w2 = loadMatrix(weightsJSON, "w2");
  w3 = loadMatrix(weightsJSON, "w3");

  println("w1 loaded with dimensions: " + w1.length + " x " + w1[0].length);
  println("w2 loaded with dimensions: " + w2.length + " x " + w2[0].length);
  println("w3 loaded with dimensions: " + w3.length + " x " + w3[0].length);
}

void draw() {
  // Draw on the high-resolution canvas
  canvas.beginDraw();
  if (mousePressed && mouseX > visualization.width) {
    canvas.stroke(255);
    canvas.strokeWeight(4); // Adjust line thickness for higher resolution
    canvas.line(
      (float)(mouseX - visualization.width) / visualization.width * canvas.width, 
      (float)mouseY / visualization.height * canvas.height, 
      (float)(pmouseX - visualization.width) / visualization.width * canvas.width, 
      (float)pmouseY / visualization.height * canvas.height
    );
  }
  canvas.endDraw();

  // Scale down canvas to 28x28 for neural network input
  PImage smallCanvas = canvas.get();
  smallCanvas.resize(28, 28);
  smallCanvas.loadPixels();

  // Convert the 28x28 scaled-down canvas to the input matrix for the neural network
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      int idx = i * 28 + j;
      inputMat[0][idx] = ((smallCanvas.pixels[idx] >> 16) & 0xFF) / 255.0;
    }
  }
  
  
  //canvas.loadPixels();

  //// Convert canvas drawing to input matrix
  //for (int i = 0; i < 28; i++) {
  //  for (int j = 0; j < 28; j++) {
  //    int idx = i * 28 + j;
  //    inputMat[0][idx] = ((canvas.pixels[idx] >> 16) & 0xFF) / 255.0;
  //  }
  //}
  //canvas.endDraw();
  
  //// Scale down canvas to 28x28 for neural network input
  //PImage smallCanvas = canvas.get();
  //smallCanvas.resize(28, 28);
  //smallCanvas.loadPixels();


  // Pass through the neural network layers
  float[][] mat1 = relu(multMat(inputMat, w1));
  float[][] mat2 = relu(multMat(mat1, w2));
  float[][] mat3 = multMat(mat2, w3);

  // Reshape for visualization
  float[][] reshapedInput = reshape(inputMat, 28);
  float[][] reshapedLayer1 = reshape(mat1, 8);
  float[][] reshapedLayer2 = reshape(mat2, 4);

  // Visualization
  visualization.beginDraw();
  visualization.background(0);

  PVector[][] inputPos  = drawMat(reshapedInput, 0, visualization);
  PVector[][] layer1Pos = drawMat(reshapedLayer1, -100, visualization);
  PVector[][] layer2Pos = drawMat(reshapedLayer2, -150, visualization);
  PVector[][] outputPos = drawResult(softmax(mat3), -200, visualization);

  if (visualizeLines) {
    drawLines(inputPos, layer1Pos, visualization);
    drawLines(layer1Pos, layer2Pos, visualization);
    drawLines(layer2Pos, outputPos, visualization);
  }

  visualization.endDraw();

  // Display the results
  background(0);
  image(visualization, 0, 0);
  image(canvas, visualization.width, 0, visualization.width, visualization.height);
  
  // Draw dividing line between the areas
  stroke(255);          // Set the line color to white
  strokeWeight(2);      // Set the line thickness
  line(visualization.width, 0, visualization.width, height); // Draw the line
  // Draw clear screen button
  drawClearButton();
}


void drawClearButton() {
  fill(50, 50, 200);
  noStroke();
  rect(visualization.width + 10, height - 40, 100, 30);
  
  fill(255);
  textAlign(CENTER, CENTER);
  text("Clear Screen", visualization.width + 60, height - 25);
}

void mousePressed() {
  // Check if the mouse is over the "Clear Screen" button
  if (mouseX > visualization.width + 10 && mouseX < visualization.width + 110 && mouseY > height - 40 && mouseY < height - 10) {
    clearCanvas();
  }
}

void clearCanvas() {
  // Clear the drawing canvas
  canvas.beginDraw();
  canvas.background(0);
  canvas.endDraw();
}

void keyPressed() {
  canvas.beginDraw();
  canvas.background(0);
  canvas.endDraw();
}

// Load matrix from JSON
float[][] loadMatrix(JSONObject json, String key) {
  JSONArray matrixArray = json.getJSONArray(key);
  int rows = matrixArray.size();
  int cols = matrixArray.getJSONArray(0).size();
  
  float[][] matrix = new float[rows][cols];
  for (int i = 0; i < rows; i++) {
    JSONArray rowArray = matrixArray.getJSONArray(i);
    for (int j = 0; j < cols; j++) {
      matrix[i][j] = rowArray.getFloat(j);
    }
  }
  return matrix;
}

// Activation function
float[][] relu(float[][] mat) {
  int rows = mat.length;
  int cols = mat[0].length;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      mat[i][j] = max(0, mat[i][j]);
    }
  }
  return mat;   //z
}

// Softmax function
float[][] softmax(float[][] x) {
  float[][] output = new float[1][x[0].length];
  float sum = 0;
  for (int i = 0; i < x[0].length; i++) {
    output[0][i] = exp(x[0][i]);
    sum += output[0][i];
  }
  for (int i = 0; i < x[0].length; i++) {
    output[0][i] /= sum;
  }
  return output;
}

// Matrix multiplication
float[][] multMat(float[][] matA, float[][] matB) {
  int rowA = matA.length;
  int colA = matA[0].length;
  int rowB = matB.length;
  int colB = matB[0].length;

  if (colA != rowB) {
    println("Error: Incompatible matrix dimensions for multiplication.");
    return null;
  }

  float[][] result = new float[rowA][colB];
  for (int i = 0; i < rowA; i++) {
    for (int j = 0; j < colB; j++) {
      for (int k = 0; k < colA; k++) {
        result[i][j] += matA[i][k] * matB[k][j];
      }
    }
  }
  return result;
}

// Reshape 1D matrix to 2D for visualization
float[][] reshape(float[][] mat, int desiredColNum) {
  int col = mat.length;
  int row = mat[0].length;
  col = desiredColNum;
  row = row / col;
  float[][] result = new float[desiredColNum][row];
  int idx = 0;
  if (row * col != mat[0].length) {
    println("Error: Cannot reshape matrix.");
    return null;
  }
  for (int i = 0; i < col; i++) {
    for (int j = 0; j < row; j++) {
      result[i][j] = mat[0][idx];
      idx++;
    }
  }
  return result;
}

// Visualization functions
PVector[][] drawMat(float[][] mat, float yPosition, PGraphics pg) {
  int row = mat.length;
  int col = mat[0].length;
  float scale = 12;
  float boxSize = 10;
  PVector[][] result = new PVector[row][col];

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      pg.pushMatrix();
      result[i][j] = new PVector(yPosition, i * scale - (row * scale) * 0.5, j * scale - (col * scale) * 0.5);
      pg.translate(result[i][j].x, result[i][j].y, result[i][j].z);
      pg.stroke(255);
      pg.fill(mat[i][j] * 255);
      pg.box(boxSize);
      pg.popMatrix();
    }
  }
  return result;
}

PVector[][] drawResult(float[][] mat, float yPosition, PGraphics pg) {
  int row = mat.length;
  int col = mat[0].length;
  float scale = 12;
  float boxSize = 10;
  PVector[][] result = new PVector[row][col];
  pg.textAlign(CENTER);
  pg.rectMode(CENTER);

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      pg.pushMatrix();
      result[i][j] = new PVector(yPosition, i * scale - (row * scale) * 0.5, j * scale - (col * scale) * 0.5);
      pg.translate(result[i][j].x, result[i][j].y, result[i][j].z);
      pg.rotateY(-HALF_PI);
      pg.stroke(255);
      pg.fill(mat[i][j] * 255);
      pg.box(boxSize);
      pg.textSize(18);
      pg.fill(0);
      pg.noStroke();
      pg.rect(0, 10, 20, 20);
      pg.fill(mat[i][j] * 255);
      pg.text(j, 0, 20, 10);
      pg.popMatrix();
    }
  }
  return result;
}

void drawLines(PVector[][] matAPos, PVector[][] matBPos, PGraphics pg) {
  int rowA = matAPos.length;
  int colA = matAPos[0].length;
  int rowB = matBPos.length;
  int colB = matBPos[0].length;

  for (int i = 0; i < rowA; i++) {  
    for (int j = 0; j < colA; j++) {
      for (int k = 0; k < rowB; k++) {  
        for (int l = 0; l < colB; l++) {
          pg.stroke(255, 100);
          pg.line(matAPos[i][j].x, matAPos[i][j].y, matAPos[i][j].z, matBPos[k][l].x, matBPos[k][l].y, matBPos[k][l].z);
        }
      }
    }
  }
}
