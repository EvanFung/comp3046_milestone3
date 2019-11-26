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
    const static int thread_count = 32;

    Matrix(){}

    Matrix(vector<vector<T>> scalers, int col, int row)
	{
		set(scalers, col, row);
	}

    Matrix(float from, float to, int col, int row)
    {
        setRandom(from, to, col, row);
    }

    Matrix(int col, int row)
    {
        setZero(col, row);
    }

	void set(vector<vector<T>> scalers, int col, int row)
	{
        size[0] = col;
        size[1] = row;
		matrixData.assign(scalers.begin(), scalers.end());
	}

    void setRandom(float from, float to, int col, int row)
    {
        size[0] = col;
        size[1] = row;
        vector<vector<T>> scalers = fillRandom(from, to);
        matrixData.assign(scalers.begin(), scalers.end());
    }

    void setZero(int col, int row)
    {
        size[0] = col;
        size[1] = row;
        vector<vector<T>> scalers = fillZero();
        matrixData.assign(scalers.begin(), scalers.end());
    }


	Matrix<T> transpose()
	{
		vector<vector<T>> tmp;

		for (int i = 0; i < this->getRowSize(); i++)
		{
            vector<T> tmpRow;
			for (int j = 0; j < this->getColSize(); j++)
			{
                tmpRow.push_back(matrixData[j][i]);
			}
            tmp.push_back(tmpRow);
		}
		Matrix<T> tmpMatrix(tmp, this->getRowSize(), this->getColSize());
		return tmpMatrix;
	}

	Matrix<T> operator * (const Matrix<T> m)
	{
        vector<vector<T>> tmp;

        #pragma omp parallel for num_threads(thread_count)
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            for (int j = 0; j < m.getRowSize(); j++)
            {
                T elem = 0;

                for (int k = 0; k < this->getRowSize(); k++)
                {
                    elem += this->getMatrix()[i][k] * m.getMatrix()[k][j];
                }

                tmpRow.push_back(elem);
            }
            tmp.push_back(tmpRow);
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), m.getRowSize());
        return tmpMatrix;
	}

    Matrix constOp (T(*operation)(T,float), T constant)
    {
        vector<vector<T>> tmp;
        #pragma omp parallel for num_threads(thread_count)
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmpRow.push_back(operation(matrixData[i][j], constant));
            }
            tmp.push_back(tmpRow);
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), this->getRowSize());
        return tmpMatrix;
    }

    Matrix operator * (const float constant){return constOp(&constMulOp, constant);}
    static T constMulOp(T tar, float constant){return tar * constant;}
    Matrix operator / (const float constant){return constOp(&constDivOp, constant);}
    static T constDivOp(T tar, float constant){return tar / constant;}

    Matrix matrixDirOp (T(*operation)(T,float), Matrix m)
    {
        vector<vector<T>> tmp;
        #pragma omp parallel for num_threads(thread_count)
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmpRow.push_back(operation(matrixData[i][j], m.matrixData[i][j]));
            }
            tmp.push_back(tmpRow);
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), this->getRowSize());
        return tmpMatrix;
    }

    Matrix operator + (const Matrix m){return matrixDirOp(&matrixAddOp, m);}
    static T matrixAddOp(T tar, float constant){return tar + constant;}

    Matrix operator - (const Matrix m){return matrixDirOp(&matrixSupOp, m);}
    static T matrixSupOp(T tar, float constant){return tar - constant;}

    vector<vector<T>> fillRandom(float from, float to)
    {
        default_random_engine randEngine(time(NULL));
        uniform_real_distribution<T> realDist(from, to);
        realDist(randEngine);

        vector<vector<T>> tmp;
        #pragma omp parallel for num_threads(thread_count)
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmpRow.push_back(realDist(randEngine));
            }
            tmp.push_back(tmpRow);
        }

        return tmp;
    }

    vector<vector<T>> fillZero()
    {
        vector<vector<T>> tmp;
#pragma omp parallel for num_threads(thread_count)
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmpRow.push_back(0.0);
            }
            tmp.push_back(tmpRow);
        }

        return tmp;
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


