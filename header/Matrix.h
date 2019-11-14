#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

template <class T>
class Matrix
{
private:

	vector<vector<T>> matrixData;
	int size[2] = { 0,0 };


public:

    Matrix(){}

    Matrix(vector<vector<T>> scalers, int col, int row)
	{
		set(scalers, col, row);
	}

	void set(vector<vector<T>> scalers, int col, int row)
	{
		size[0] = col;
		size[1] = row;
		matrixData.assign(scalers.begin(), scalers.end());
	}

	Matrix<T> transpose()
	{
		vector<vector<T>> tmp;

		for (int i = 0; i < this->getRowSize(); i++)
		{
			tmp.push_back(*new vector<T>);
			for (int j = 0; j < this->getColSize(); j++)
			{
				tmp[i].push_back(matrixData[j][i]);
			}
		}
		Matrix<T> tmpMatrix(tmp, this->getRowSize(), this->getColSize());
		return tmpMatrix;
	}

	Matrix<T> operator * (const Matrix<T> m)
	{
        vector<vector<T>> tmp;

        for (int i = 0; i < this->getColSize(); i++)
        {
            tmp.push_back(*new vector<T>);
            for (int j = 0; j < m.getRowSize(); j++)
            {
                T elem = 0;

                for (int k = 0; k < this->getRowSize(); k++)
                {
                    elem += this->getMatrix()[i][k] * m.getMatrix()[k][j];
                }

                tmp[i].push_back(elem);
            }
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), m.getRowSize());
        return tmpMatrix;
	}

	Matrix operator * (const float mul)
	{
		vector<vector<T>> tmp;

		for (int i = 0; i < this->getColSize(); i++)
		{
			tmp.push_back(*new vector<T>);
			for (int j = 0; j < this->getRowSize(); j++)
			{
				tmp[i].push_back(matrixData[i][j] * mul);
			}
		}
		Matrix<T> tmpMatrix(tmp, this->getColSize(), this->getRowSize());
		return tmpMatrix;
	}

	Matrix operator + (const Matrix m)
	{
        vector<vector<T>> tmp;

        for (int i = 0; i < this->getColSize(); i++)
        {
            tmp.push_back(*new vector<T>);
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmp[i].push_back(matrixData[i][j] + m.matrixData[i][j]);
            }
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), this->getRowSize());
        return tmpMatrix;
	}

    int getColSize() const
    {
        return size[0];
    }

    int getRowSize() const
    {
        return size[1];
    }

	vector<vector<T>> getMatrix() const
    {
		return matrixData;
	}

	void print()
	{
		for (int i = 0; i < this->getColSize(); i++)
		{
			for (int j = 0; j < this->getRowSize(); j++)
			{
			    printf("%10.4f", matrixData[i][j]);
			}
			cout << endl;
		}

	}

};


