#include <iostream>
#include "NeuralNetwork.cpp"

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

    Matrix<float> testMatrix({{1,2,3,4,5,6,7,8,9}}, 1,9);
    testMatrix = testMatrix.transpose();

    Matrix<float> gTMatrix({{1,0,1}}, 1,3);
    gTMatrix = gTMatrix.transpose();

    NeuralNetwork<float> neuralNetwork(testMatrix, gTMatrix,{8,6});

    cout << "neuralNetwork.outputs(): " << endl;
    neuralNetwork.print(neuralNetwork.outputs());
    cout << endl;
    cout << "neuralNetwork.errors(neuralNetwork.outputs()): " << endl;
    neuralNetwork.print(neuralNetwork.errors(neuralNetwork.outputs()));
    cout << endl;
    cout << "neuralNetwork.loss(): " << neuralNetwork.loss() << endl;
    cout << endl;

    for (int i = 0; i < 1000; ++i)
    {
        neuralNetwork.update();
    }

    cout << "neuralNetwork.outputs(): " << endl;
    neuralNetwork.print(neuralNetwork.outputs());
    cout << endl;
    cout << "neuralNetwork.errors(neuralNetwork.outputs()): " << endl;
    neuralNetwork.print(neuralNetwork.errors(neuralNetwork.outputs()));
    cout << endl;
    cout << "neuralNetwork.loss(): " << neuralNetwork.loss() << endl;
    cout << endl;
    return 0;

}