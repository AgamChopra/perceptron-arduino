float sigmoid(const float x) {
  return 1 / (1 + exp(-x));
};

float Loss(const float y, const float yp) {
  //return pow(y - yp, 2) / 2.0; //L2 aka MSE
  return abs(y - yp); //L1
};

class perceptron {
  private:
    float b = 0.0;
    float w[3] = {};
    float grad_b = 0.0;
    float grad_w[3] = {};
    float epsilon = 1E-7;
    float alpha = 1E-2;
  public:
    perceptron() {
      w[0] = random(0, 1);
      w[1] = random(0, 1);
      w[2] = random(0, 1);
    };

    void show_values() {
      Serial.println("w[]:");
      Serial.println(w[0], 6);
      Serial.println(w[1], 6);
      Serial.println(w[2], 6);
    };

    float forward(const float x[3]) {
      return sigmoid(w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b);
    };

    void backward(const float a[], const float y, const float yp) {
      float dJ = ((Loss(y, yp + epsilon)) - Loss(y, yp - epsilon)) / (2 * epsilon);
      grad_w[0] = dJ * (sigmoid(a[0] * (w[0] + epsilon) + b) - sigmoid(a[0] * (w[0] - epsilon) + b)) / (2 * epsilon);
      grad_w[1] = dJ * (sigmoid(a[1] * (w[1] + epsilon) + b) - sigmoid(a[1] * (w[1] - epsilon) + b)) / (2 * epsilon);
      grad_w[2] = dJ * (sigmoid(a[2] * (w[2] + epsilon) + b) - sigmoid(a[2] * (w[2] - epsilon) + b)) / (2 * epsilon);
      grad_b = dJ * (sigmoid(a[0] * w[0] + (b + epsilon)) - sigmoid(a[0] * w[0] + (b - epsilon))) / (2 * epsilon);
    };

    void update_param() {
      b = b - alpha * grad_b;
      w[0] = w[0] - alpha * grad_w[0];
      w[1] = w[1] - alpha * grad_w[1];
      w[2] = w[2] - alpha * grad_w[2];
    };
};

void setup() {
  Serial.begin(9600);
  Serial.println("Serial Port Connected");
  Serial.println("with Arduino Nano, running Perceptron");
};

perceptron p1;
float x[3];
float y = 0.0;
float yp = 0.0;

void loop() {
  x[0] = random(-10.0,10.0);
  x[1] = random(-10.0,10.0);
  x[2] = random(-10.0,10.0);
  if (abs((sin(x[0])+cos(x[1]))/exp(-x[2])) >= 5.0) {
    y = 1.0;
  }
  else {
    y = 0.0;
  }
  Serial.println(abs((sin(x[0])+cos(x[1]))/exp(-x[2])));
  p1.show_values();
  Serial.println("---");
  Serial.println(y);
  yp = p1.forward(x);
  Serial.println(yp);
  p1.backward(x, y, yp);
  p1.update_param();
  Serial.println("###");
};
