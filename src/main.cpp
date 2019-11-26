#include <iostream>
#include "NeuralNetwork.cpp"
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

template <class T>
void dataloader(vector<Matrix<T>> &, vector<Matrix<T>> &, string);


int main() {
    std::cout << "Hello, World!" << std::endl;

//    Matrix<float> m1({{5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}}, 2, 4);
//    Matrix<float> m2({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}}, 3, 4);
//
//    Matrix<float> m3({{5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}, {9.0, 10.0, 11.0, 12.0}}, 3, 4);
//    Matrix<float> m4({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}}, 3, 4);
//    m2.transpose().print();
//    cout << endl;
//    (m1*m2.transpose()).print();
//    cout << endl;
//    (matrixXMatrixT(m1,m2)).print();
//    cout << endl;
//
//
//    (m3.transpose()*m4).print();
//    cout << endl;
//    (matrixTXMatrix(m3,m4)).print();
//    cout << endl;

    vector<Matrix<float>> testMatrices;
    for (int i = 0; i < 10; ++i)
    {
        Matrix<float> test(-0.5,0.5,9,1);
        testMatrices.push_back(test);
    }



    vector<Matrix<float>> gTMatrices;
    {
        for (int i = 0; i < 10; ++i) {
            vector<vector<float>> tmpYCol;
            for (int j = 0; j < 10; j++)
            {
                vector<float> tmpYRow;
                tmpYRow.push_back((j == i)? 1:0);
                tmpYCol.push_back(tmpYRow);
            }
            Matrix<float> test(tmpYCol, 10,1);
            gTMatrices.push_back(test);
        }
    }

    vector<Matrix<float>> X_train;
    vector<Matrix<float>> Y_train;
    vector<Matrix<float>> X_test;
    vector<Matrix<float>> Y_test;
    dataloader(X_train, Y_train, "../data/train_small.txt");
    //dataloader(X_test, Y_test, "../data/test.txt");

    NeuralNetwork<float> neuralNetwork(X_train, Y_train, X_train, Y_train, {10}, 10,0.1);

    neuralNetwork.train();

    //neuralNetwork.update()

    //neuralNetwork.update();



    return 0;

}

template <class T>
void dataloader(vector<Matrix<T>> &X_train, vector<Matrix<T>> &Y_train, string filePath)
{
    ifstream myfile(filePath);

    if (myfile.is_open())
    {
        cout << "Loading data ...\n";
        string line;
        vector<T> Y_trainWithNum;

        while (getline(myfile, line))
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


        myfile.close();


        cout << "Loading data finished.\n";
    }
    else
        cout << "Unable to open file" << '\n';

    return;
}