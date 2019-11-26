//
// Created by badaeib on 2019年11月14日.
//

#include <iostream>
#include <ctime>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>
#include <omp.h>
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
    float learningRate = 0.01;

    vector<Matrix<T>> inputLayers;
    vector<Matrix<T>> groundTruths;

    vector<Matrix<T>> testInputs;
    vector<Matrix<T>> testGroundTruths;
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

    //inputLayers take column Matrix.
    NeuralNetwork(vector<Matrix<T>> inputLayers, vector<Matrix<T>> groundTruths, vector<Matrix<T>> testInputs, vector<Matrix<T>> testGroundTruths, vector<int> hiddenLayersNums, int batchSize = 100, T learningRate = 0.01)
    {

        this->layersNums.push_back(inputLayers[0].getColSize());
        this->layersNums.insert(this->layersNums.end(), hiddenLayersNums.begin(), hiddenLayersNums.end());
        this->layersNums.push_back(groundTruths[0].getColSize());
        iniWeightsAndBiases();
        this->inputLayers = inputLayers;
        this->groundTruths =  groundTruths;

        this->testInputs = testInputs;
        this->testGroundTruths = testGroundTruths;

        this->batchSize = batchSize;
        this->learningRate = learningRate;
        // this->activationFunction = &sigmoid;

    }

    Matrix<T> lastError(Matrix<T> output, Matrix<T> groundTruth)
    {
        return hadamardX((output - groundTruth), activationD(output));
    }

    vector<Matrix<T>> outputs(Matrix<T> inputLayer)
    {
        Matrix<T> output = inputLayer;
        vector<Matrix<T>> outputs;
        outputs.push_back(output);

        for (int i = 0; i < layersNums.size() - 1; i++)
        {
            output = layerOutput(i, output);
            outputs.push_back(output);
        }
        return outputs;
    }

    T loss(vector<Matrix<T>> testInputs, vector<Matrix<T>> testGroundTruths)
    {
        clock_t start = clock();
        cout << "loss start: " << endl;
        cout << "testGroundTruths.size(): " << testGroundTruths.size() << endl;
        T loss = 0;

        for (int i = 0; i < testInputs.size(); i++)
        {
            T singleLoss = 0;
            vector<Matrix<T>> _outputs = outputs(testInputs[i]);
            for (int j = 0; j < testGroundTruths[i].getColSize(); ++j)
            {
                singleLoss += pow(testGroundTruths[i].getMatrix()[j][0] - _outputs[_outputs.size() - 1].getMatrix()[j][0], 2.0f);
            }
            singleLoss /= 2;
            loss += singleLoss;
        }
        loss /= testInputs.size();
        clock_t end = clock();

        cout << "loss take: " << (end - start) / (double) CLOCKS_PER_SEC << " sec." << endl;
        return loss;
    }

    void update(vector<Matrix<T>> batchInputLayer, vector<Matrix<T>> batchGroundTruths)
    {
        clock_t start = clock();
        cout << "update start: " << endl;

        vector<Matrix<T>> batchWeightGradient;
        vector<Matrix<T>> batchBiasGradient;

        for (int k = 0; k < layersNums.size() - 1; ++k)
        {
            Matrix<T> emptyWeight(weights[k].getColSize(),weights[k].getRowSize());
            Matrix<T> emptyBias(biases[k].getColSize(),biases[k].getRowSize());

            batchWeightGradient.push_back(emptyWeight);
            batchBiasGradient.push_back(emptyBias);
        }

        for (int i = 0; i < batchInputLayer.size(); ++i)
        {
            vector<Matrix<T>> _outputs = outputs(batchInputLayer[i]);
            vector<Matrix<T>> _errors = errors(_outputs, batchGroundTruths[i]);

            for (int j = 0; j < layersNums.size() - 1; ++j)
            {

                batchWeightGradient[j] = batchWeightGradient[j] + (matrixXMatrixT(_errors[j], _outputs[j]));
                batchBiasGradient[j] = batchBiasGradient[j] + _errors[j];

            }
        }

        for (int l = 0; l < layersNums.size() - 1; ++l)
        {
            weights[l] = weights[l] - batchWeightGradient[l] * learningRate /  batchInputLayer.size();
            biases[l] = biases[l] - batchBiasGradient[l] * learningRate /  batchInputLayer.size();
        }

        clock_t end = clock();

        cout << "update take: " << (end - start) / (double) CLOCKS_PER_SEC << " sec." << endl;

    }

    vector<Matrix<T>> errors(vector<Matrix<T>> outputs, Matrix<T> groundTruth)
    {
        vector<Matrix<T>> errors;

        Matrix<T> errorTmp = lastError(outputs[outputs.size()-1], groundTruth);
        errors.push_back(errorTmp);

        for (int i = layersNums.size() - 2; i > 0; --i)
        {
            errorTmp = layerError(weights[i], errorTmp, outputs[i]);
            errors.insert(errors.begin(), errorTmp);
        }
        return errors;

    }

    void train()
    {
        vector<int> index;
        for (int i = 0; i < groundTruths.size(); ++i)
        {
            index.push_back(i);
        }

        default_random_engine randEngine(time(NULL));
        shuffle(begin(index), end(index), randEngine);
        cout << "Training start:" << endl;
        for (int i = 0; i < groundTruths.size() / batchSize + 1; ++i)
        {
            vector<Matrix<T>> batchInputLayer;
            vector<Matrix<T>> batchGroundTruths;

            int diff = (batchSize * i) - groundTruths.size();

            cout << "(int)index.size(): " << (int)index.size() << endl;
            for (int j = 0; j < min((int)index.size(), (int)(diff > 0 ? diff: batchSize)); ++j)
            {
                batchInputLayer.push_back(inputLayers[index[i * groundTruths.size() / batchSize + 1 + j]]);
                batchGroundTruths.push_back(groundTruths[index[i * groundTruths.size() / batchSize + 1 + j]]);
            }

            cout << endl;

            update(batchInputLayer, batchGroundTruths);
            cout << "epoch " << i << " loss: " << loss(testInputs, testGroundTruths) << endl;

        }
        cout << "Training ended." << endl;
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
        cout << "inputLayers: " << endl;
        inputLayers[0].print();
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
