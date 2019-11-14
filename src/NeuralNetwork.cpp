//
// Created by badaeib on 2019年11月14日.
//

#include <iostream>
#include <ctime>
#include <cmath>
#include <random>
#include <functional>
#include "../header/MatrixVector.h"

using namespace std;

template <class T>
class NeuralNetwork
{
private:
    vector<int> layersNums; //num of neuron for each layer, include input layer.
    vector<Matrix<T>> weights;
    vector<Matrix<T>> biases;

    Matrix<T> inputLayer;
    //T ( NeuralNetwork::*activationFunction ) ( T );

    void iniWeightsAndBiases()
    {
        default_random_engine randEngine(time(NULL));
        uniform_real_distribution<T> realDist(-0.1, 0.1);
        realDist(randEngine);

        //loop through all layers.
        for (int i = 0; i < layersNums.size() - 1; i++)
        {
            vector<vector<T>> weightsTmp;
            vector<vector<T>> biasesTmp;

            //loop through nodes of each layers.
            for (int j = 0; j < layersNums[i + 1]; j++)
            {
                weightsTmp.push_back(*new vector<T>);
                biasesTmp.push_back(*new vector<T>);

                biasesTmp[j].push_back({realDist(randEngine)});

                //loop through each connection for a node.
                for (int k = 0; k < layersNums[i]; k++) {
                    weightsTmp[j].push_back(realDist(randEngine));
                }
            }

            weights.push_back(*new Matrix<float>(weightsTmp, layersNums[i + 1], layersNums[i]));
            biases.push_back(*new Matrix<float>(biasesTmp, layersNums[i + 1], 1));
        }
    }

public:


    //inputLayer take column Matrix.
    NeuralNetwork(Matrix<T> inputLayer, vector<int> layersNums)
    {
        this->layersNums.push_back(inputLayer.getColSize());
        this->layersNums.insert( this->layersNums.end(), layersNums.begin(), layersNums.end());
        iniWeightsAndBiases();
        this->inputLayer = inputLayer;
        //this->activationFunction = &sigmoid;

    }

    T sigmoid (T x)
    {
        return 1.0/(1.0+ pow(M_E,-x));
    };

    Matrix<T> feed()
    {
        Matrix<T> layer = inputLayer;
        for (int i = 0; i < layersNums.size() - 1; i++)
        {
            layer = feedforward(i, layer);
        }
        return layer;
    }

    Matrix<T> feedforward(int layer, Matrix<T> preLayer)
    {
        Matrix tmp =  weights[layer] * preLayer + biases[layer];
        vector<vector<T>> sigTmp;
        for (int i = 0; i < tmp.getColSize(); i++)
        {
            sigTmp.push_back(*new vector<T>);
            sigTmp[i].push_back(sigmoid(tmp.getMatrix()[i][0]));
        }

        return *new Matrix<T>(sigTmp, tmp.getColSize(), 1);
    }

    void weightsDebug()
    {
        cout << "weights: " << endl;
        for (int i = 0; i < weights.size(); ++i)
        {
            weights[i].print();
            cout << endl;
        }
    }

    void biasesDebug()
    {
        cout << "biases: " << endl;
        for (int i = 0; i < weights.size(); ++i)
        {
            biases[i].print();
            cout << endl;
        }
    }

    void inputDebug()
    {
        cout << "inputLayer: " << endl;
        inputLayer.print();
        cout << endl;
    }
};
