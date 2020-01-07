// MLP_IrisDataset.cpp : º×²ÉÀÇ ºÐ·ù MLP ÇÐ½À
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <array>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>

using namespace std;

vector<vector<float>> training_data;
vector<float> a = {0, 1, 2};
float learning_rate = 1.0;
float momentum = 0.25;
float RMSE_array_error[20000], update_w[25], w[25], error[3], gradients[25], user_input[4];
int bias = 1;
vector<float> prev_weight_update = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
float RMSE_ERROR = 1;
float h1, h2, h3, h4, output_neuron, derivative_O1, derivative_h1, derivative_h2, derivative_h3, derivative_h4, sum_output, sum_h1, sum_h2, sum_h3, sum_h4;
int epoch = 0;
char choise = 'Y';
bool repeat = false;

float sigmoid_function(float x);
void readcsv();
void calc_hidden_layers(int x);
void calc_output_neuron();
void calc_error(int x);
void calc_derivatives(int x);
void calc_gradient(int x);
void calc_updates();
void update_new_w();
void calc_RMSE_ERROR();
void generate_w();
void train_neural_net();
void start_input();
void safe_data();

int main()
{
	readcsv();
	generate_w();
	train_neural_net();
	safe_data();
	start_input();
	system("pause");
}

void readcsv() {
	ifstream data("C:\\Users\\kk\\Documents\\Visual Studio 2015\\Projects\\MLP_IrisDataset\\dataset\\iris DA.csv");
	string line, field;
	vector<string> v;
	vector<vector<string>> training_data;
	for (size_t i = 0; i < 1; i++)
		getline(data, line);
	while (getline(data, line)) {
		v.clear();
		stringstream ss(line);
		for(size_t i=0; i<4; i++) {
			getline(ss, field, ','); 
			v.push_back(field);
		}
		training_data.push_back(v);
	}
}

 void safe_data()
 {
 	ofstream dataER;
 	dataER.open("errorData1.txt");
 	for (int i = 0; i < epoch; i++)
 	{
 		dataER << i << "   " << RMSE_array_error[i] << endl;
 	}
 	dataER.close();
 
 	ofstream dataER1;
 	dataER1.open("weight_data1.txt");
 	for (int i = 0; i < 25; i++)
 	{
 		dataER1 << i << "   " << w[i] << endl;
 	}
 	dataER1.close();
 }
 
 void start_input()
 {
 	do
 	{
 		if (choise == 'Y')
 		{
 			cout << "enter Sepal Length: "; cin >> user_input[0]; cout << endl;
 			cout << "enter Sepal Width: "; cin >> user_input[1]; cout << endl;
			cout << "enter Petal Length: "; cin >> user_input[2]; cout << endl;
			cout << "enter Petal Width: "; cin >> user_input[3]; cout << endl;
 			sum_h1 = (user_input[0] * w[0]) + (user_input[1] * w[4]) + (user_input[2] * w[8]) + (user_input[3] * w[12]) + (bias * w[16]);
 			sum_h2 = (user_input[0] * w[1]) + (user_input[1] * w[5]) + (user_input[2] * w[9]) + (user_input[3] * w[13]) + (bias * w[17]);
			sum_h3 = (user_input[0] * w[2]) + (user_input[1] * w[6]) + (user_input[2] * w[10]) + (user_input[3] * w[14]) + (bias * w[18]);
			sum_h4 = (user_input[0] * w[3]) + (user_input[1] * w[7]) + (user_input[2] * w[11]) + (user_input[3] * w[15]) + (bias * w[19]);
 			h1 = sigmoid_function(sum_h1);
 			h2 = sigmoid_function(sum_h2);
			h3 = sigmoid_function(sum_h3);
			h4 = sigmoid_function(sum_h4);
 
 			sum_output = (h1 * w[20]) + (h2 * w[21]) + (h3 * w[22]) + (h4 * w[23]) + (bias * w[24]);
 			output_neuron = sigmoid_function(sum_output);

			if (0 < output_neuron < 0.5)
				cout << "output = setosa" << endl;
			else if (0.5 < output_neuron < 1.5)
				cout << "output = versicolor" << endl;
			else if (1.5 < output_neuron < 2.5)
				cout << "output = virginica" << endl;

 			cout << "Again? Y/N"; cin >> choise;
 		}
 		else
 		{
 			break;
 		}
 	} while ((choise == 'Y' || 'y') && (choise != 'n' || 'N'));
 }
 
 void train_neural_net()
 {
 	while (epoch < 20000)
 	{
 		for (int i = 0; i < 3; i++)
 		{
 			calc_hidden_layers(i);
 			calc_output_neuron();
 			calc_error(i);
 			calc_derivatives(i);
 			calc_gradient(i);
 			calc_updates();
 			update_new_w();
 		}
 		calc_RMSE_ERROR();
 		RMSE_array_error[epoch] = RMSE_ERROR;
 		cout << "epoch: " << epoch << endl;
 		epoch = epoch + 1;
 
 		//Adding some motivation so if the neural network is not converging after 4000 epochs it will start over again until it converges
 		if (epoch > 4000 && RMSE_ERROR > 0.5)
 		{
 			repeat = true;
 			for (int i = 0; i < 25; i++)
 			{
 				prev_weight_update[i] = 0;
 				update_w[i] = 0;
 				gradients[i] = 0;
 			}
 			for (int i = 0; i < 3; i++)
 				error[i] = 0;
 			for (int i = 0; i < epoch; i++)
 				RMSE_array_error[i] = 0;
 			epoch = 0;
 			generate_w();
 		}
 	}
 }
 
 float sigmoid_function(float x)
 {
 	float sigmoid = 1 / (1 + exp(-x));
 	return sigmoid;
 }
 
 void generate_w()
 {
 	srand(time(NULL));
 	for (int i = 0; i < 25; i++)
 	{
 		int randNum = rand() % 2;
 		if (randNum == 1)
 			w[i] = -1 * (double(rand()) / (double(RAND_MAX) + 1.0)); // generate number between -1.0 and 0.0
 		else
 			w[i] = double(rand()) / (double(RAND_MAX) + 1.0); // generate number between 1.0 and 0.0
 
 		cout << "weight " << i << " = " << w[i] << endl;
 	}
 	cout << "" << endl;
 }
 
 
 void calc_hidden_layers(int x)
 {
 	sum_h1 = (training_data[x][0] * w[0]) + (training_data[x][1] * w[4]) + (training_data[x][2] * w[8])+ (training_data[x][3] * w[12]) + (bias * w[16]);
 	sum_h2 = (training_data[x][0] * w[1]) + (training_data[x][1] * w[5]) + (training_data[x][2] * w[9]) + (training_data[x][3] * w[13]) + (bias * w[17]);
	sum_h3 = (training_data[x][0] * w[2]) + (training_data[x][1] * w[6]) + (training_data[x][2] * w[10]) + (training_data[x][3] * w[14]) + (bias * w[18]);
	sum_h4 = (training_data[x][0] * w[3]) + (training_data[x][1] * w[7]) + (training_data[x][2] * w[11]) + (training_data[x][3] * w[15]) + (bias * w[19]);
 	h1 = sigmoid_function(sum_h1);
 	h2 = sigmoid_function(sum_h2);
	h3 = sigmoid_function(sum_h3);
	h4 = sigmoid_function(sum_h4);
 }
 
 void calc_output_neuron()
 {
 	sum_output = (h1 * w[20]) + (h2 * w[21]) + (h3 * w[22]) + (h4 * w[23]) + (bias * w[24]);
 	output_neuron = sigmoid_function(sum_output);
 }
 
 void calc_error(int x)
 {
 	error[x] = output_neuron - a[x];
 }
 
 void calc_derivatives(int x)
 {
 	derivative_O1 = -error[x] * (exp(sum_output) / pow((1 + exp(sum_output)), 2));
 	derivative_h1 = (exp(sum_h1) / pow((1 + exp(sum_h1)), 2)) * w[20] * derivative_O1;
 	derivative_h2 = (exp(sum_h2) / pow((1 + exp(sum_h2)), 2)) * w[21] * derivative_O1;
	derivative_h3 = (exp(sum_h3) / pow((1 + exp(sum_h3)), 2)) * w[22] * derivative_O1;
	derivative_h4 = (exp(sum_h4) / pow((1 + exp(sum_h4)), 2)) * w[23] * derivative_O1;
 }
 
 void calc_gradient(int x)
 {
 	gradients[0] = sigmoid_function(training_data[x][0]) * derivative_h1;
 	gradients[1] = sigmoid_function(training_data[x][1]) * derivative_h1;
 	gradients[2] = sigmoid_function(training_data[x][2]) * derivative_h1;
 	gradients[3] = sigmoid_function(training_data[x][3]) * derivative_h1;
	gradients[4] = sigmoid_function(training_data[x][0]) * derivative_h2;
	gradients[5] = sigmoid_function(training_data[x][1]) * derivative_h2;
	gradients[6] = sigmoid_function(training_data[x][2]) * derivative_h2;
	gradients[7] = sigmoid_function(training_data[x][3]) * derivative_h2;
	gradients[8] = sigmoid_function(training_data[x][0]) * derivative_h3;
	gradients[9] = sigmoid_function(training_data[x][1]) * derivative_h3;
	gradients[10] = sigmoid_function(training_data[x][2]) * derivative_h3;
	gradients[11] = sigmoid_function(training_data[x][3]) * derivative_h3;
	gradients[12] = sigmoid_function(training_data[x][0]) * derivative_h4;
	gradients[13] = sigmoid_function(training_data[x][1]) * derivative_h4;
	gradients[14] = sigmoid_function(training_data[x][2]) * derivative_h4;
	gradients[15] = sigmoid_function(training_data[x][3]) * derivative_h4;
 	gradients[16] = sigmoid_function(bias) * derivative_h1;
 	gradients[17] = sigmoid_function(bias) * derivative_h2;
	gradients[18] = sigmoid_function(bias) * derivative_h3;
	gradients[19] = sigmoid_function(bias) * derivative_h4;
 	gradients[20] = h1 * derivative_O1;
 	gradients[21] = h2 * derivative_O1;
	gradients[22] = h3 * derivative_O1;
	gradients[23] = h4 * derivative_O1;
 	gradients[24] = sigmoid_function(bias) * derivative_O1;
 }
 
 void calc_updates()
 {
 	for (int i = 0; i < 25; i++)
 	{
 		update_w[i] = (learning_rate * gradients[i]) + (momentum * prev_weight_update[i]);
 		prev_weight_update[i] = update_w[i];
 	}
 }
 
 void update_new_w()
 {
 	for (int i = 0; i < 25; i++)
 	{
 		w[i] = w[i] + update_w[i];
 	}
 }
 
 void calc_RMSE_ERROR()
 {
 	RMSE_ERROR = sqrt((pow(error[0], 2) + pow(error[1], 2) + pow(error[2], 2) / 3));
 	cout << "RMSE error: " << RMSE_ERROR << endl;
 	cout << "" << endl;
 }
