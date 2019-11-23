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
    int batchSize = 100;

    Matrix<T> inputLayer;
    Matrix<T> goundTruth;
    //T (*activationFunction) ( T );

    void iniWeightsAndBiases()
    {
        //loop through all layers.
        for (int i = 0; i < layersNums.size() - 1; i++)
        {
            Matrix<T> weight(-1, 1, layersNums[i + 1], layersNums[i]);
            Matrix<T> bias(-1, 1, layersNums[i + 1], 1);
            weights.push_back(weight);
            biases.push_back(bias);
        }
    }

    Matrix<T> layerOutput(int layerIndex, Matrix<T> preLayer)
    {
        Matrix actives = activation(weights[layerIndex] * preLayer + biases[layerIndex]);
        return actives;
    }



    Matrix<T> layerError(Matrix<T> postWeights, Matrix<T> postError, Matrix<T> output)
    {
        return hadamardX(matrixTXMatrix(postWeights, postError), activationD(output));
    }

    static T sigmoid (T x)
    {
        return 1.0/(1.0+ pow(M_E,-x));
    }
    static T sigmoidD (T sig)
    {
        return sig * (1.0-sig);
    }

    Matrix<T> activation(Matrix<T> layer)
    {
        vector<vector<T>> activesTmp;
        for (int i = 0; i < layer.getColSize(); i++)
        {
            vector<T> activeTmp;
            activeTmp.push_back(sigmoid(layer.getMatrix()[i][0]));
            activesTmp.push_back(activeTmp);
        }

        Matrix<T> actives(activesTmp, layer.getColSize(), 1);
        return actives;
    }

    Matrix<T> activationD(Matrix<T> layer)
    {
        vector<vector<T>> activesTmp;
        for (int i = 0; i < layer.getColSize(); i++)
        {
            vector<T> activeTmp;
            activeTmp.push_back(sigmoidD(layer.getMatrix()[i][0]));
            activesTmp.push_back(activeTmp);
        }

        Matrix<T> actives(activesTmp, layer.getColSize(), 1);
        return actives;
    }

public:

    //inputLayer take column Matrix.
    NeuralNetwork(Matrix<T> inputLayer, Matrix<T> goundTruth, vector<int> hiddenLayersNums, int batchSize = 100)
    {
        this->layersNums.push_back(inputLayer.getColSize());
        this->layersNums.insert(this->layersNums.end(), hiddenLayersNums.begin(), hiddenLayersNums.end());
        this->layersNums.push_back(goundTruth.getColSize());
        iniWeightsAndBiases();
        this->inputLayer = inputLayer;
        this->goundTruth =  goundTruth;
        this->batchSize = batchSize;
        // this->activationFunction = &sigmoid;

    }

    Matrix<T> lastError(Matrix<T> output)
    {
        return hadamardX((output - goundTruth), activationD(output));
    }

    vector<Matrix<T>> outputs()
    {
        Matrix<T> output = inputLayer;
        vector<Matrix<T>> outputs;
        outputs.push_back(inputLayer);

        for (int i = 0; i < layersNums.size() - 1; i++)
        {
            output = layerOutput(i, output);
            outputs.push_back(output);
        }
        return outputs;
    }

    T loss()
    {
       T loss = 0;

        for (int i = 0; i < goundTruth.getColSize(); i++)
        {
            loss += pow(goundTruth.getMatrix()[i][0] - outputs()[outputs().size() - 1].getMatrix()[i][0], 2.0f);
        }
        return loss/2;
    }

    void update()
    {
        vector<Matrix<T>> _outputs = outputs();
        vector<Matrix<T>> _errors = errors(_outputs);
        for (int i = 0; i < layersNums.size() - 1; ++i)
        {
            weights[i] = weights[i] - matrixXMatrixT(_errors[i], _outputs[i]);
            biases[i] = biases[i] - _errors[i];
        }
    }

//    vector<vector<Matrix<T>>> batchErrors()
//    {
//        vector<vector<Matrix<T>>> _batchErrors;
//    }

    vector<Matrix<T>> errors(vector<Matrix<T>> outputs)
    {
        vector<Matrix<T>> errors;

        Matrix<T> errorTmp = lastError(outputs[outputs.size()-1]);
        errors.push_back(errorTmp);

        for (int i = layersNums.size() - 2; i > 0; --i)
        {
            errorTmp = layerError(weights[i], errorTmp, outputs[i]);
            errors.insert(errors.begin(), errorTmp);
        }
        return errors;

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

    void print(vector<Matrix<T>> matrices)
    {
        for (int i = 0; i < matrices.size(); ++i)
        {
            matrices[i].print();
            cout << endl;
        }
    }
};
