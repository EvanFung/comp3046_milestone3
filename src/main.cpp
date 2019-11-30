#include <iostream>
#include "NeuralNetwork.cpp"
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>

using namespace std;

template <class T>
void dataLoader(vector<Matrix<T>> &, vector<Matrix<T>> &, string);

int main() {

    vector<Matrix<float>> X_train;
    vector<Matrix<float>> Y_train;
    vector<Matrix<float>> X_test;
    vector<Matrix<float>> Y_test;
    dataLoader(X_train, Y_train, "../data/train.txt");
    dataLoader(X_test, Y_test, "../data/test.txt");

    NeuralNetwork<float> neuralNetwork(X_train, Y_train, X_test, Y_test, {100,50,25});

    neuralNetwork.train(10,128,1);
    neuralNetwork.save("save", "train");
    neuralNetwork.load("save/train.ann");
    neuralNetwork.train(10,128,1);
    neuralNetwork.save("save", "train");
    return 0;

}


template <class T>
void dataLoader(vector<Matrix<T>> &X_train, vector<Matrix<T>> &Y_train, string filePath)
{
    ifstream myFile(filePath);

    if (myFile.is_open())
    {
        cout << "Loading data ...\n";
        string line;
        vector<T> Y_trainWithNum;

        while (getline(myFile, line))
        {
            int x, y;
            stringstream ss(line);
            ss >> y;
            Y_trainWithNum.push_back(y);

            vector<vector<T>> tmpXCol;
            for (int i = 0; i < 28 * 28; i++) {
                ss >> x;
                vector<T> tmpXRow;
                tmpXRow.push_back(x/255.0);
                tmpXCol.push_back(tmpXRow);
            }
            Matrix<T> tmpX(tmpXCol, tmpXCol.size(),1);
            X_train.push_back(tmpX);
        }

        for (int k = 0; k < Y_trainWithNum.size(); ++k)
        {
            vector<vector<T>> tmpYCol;
            for (int j = 0; j < 10; j++)
            {
                vector<T> tmpYRow;
                tmpYRow.push_back((j == Y_trainWithNum[k])? 1:0);
                tmpYCol.push_back(tmpYRow);
            }
            Matrix<T> tmpY(tmpYCol, tmpYCol.size(),1);
            Y_train.push_back(tmpY);
        }


        myFile.close();


        cout << "Loading data finished.\n";
    }
    else
        cout << "Unable to open file" << '\n';

    return;
}