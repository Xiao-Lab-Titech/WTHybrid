#ifndef _RAWVECTOR_H_
#define _RAWVECTOR_H_

#include "Solver.H"

template <typename T>
void initVector1d(T*& v, int n, T val) {
    v = new T[n];

    for (unsigned int i = 0; i < n; i++) {
        v[i] = val;
    }
}

// called when T is scalar (e.g. int, double and float)
template <typename T, typename std::enable_if<std::is_scalar<T>::value, std::nullptr_t>::type = nullptr>
void initVector2d(T**& v, int a, int b, const T val) {
    v = new T*[a];
    for (unsigned int i = 0; i < a; i++) {
        v[i] = new T[b];
    }

    for (unsigned int i = 0; i < a; i++) {
        for (unsigned int j = 0; j < b; j++) {
            v[i][j] = val;
        }
    }
}

// called when T is class, struct
template <typename T, typename std::enable_if<std::is_class<T>::value, std::nullptr_t>::type = nullptr>
void initVector2d(T**& v, int a, int b, const T& val) {
    v = new T*[a];
    for (int i = 0; i < a; i++) {
        v[i] = new T[b];
    }

    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            v[i][j] = val;
        }
    }
}

template <typename T>
void initVector3d(T***& v, int a, int b, int c, T val) {
    v = new T**[a];
    for (unsigned int i = 0; i < a; i++) {
        v[i] = new T*[b];
        for (unsigned int j = 0; j < b; j++) {
            v[i][j] = new T[c];
        }
    }

    for (unsigned int i = 0; i < a; i++) {
        for (unsigned int j = 0; j < b; j++) {
            for (unsigned int k = 0; k < c; k++) {
                v[i][j][k] = val;
            }
        }
    }
}

template <typename T>
void freeVector1d(T*& v) {
    delete[] v;
    v = nullptr;
}

template <typename T>
void freeVector2d(T**& v, int a) {
    for (unsigned int i = 0; i < a; i++) {
        delete[] v[i];
    }
    delete[] v;
    v = nullptr;
}

template <typename T>
void freeVector3d(T***& v, int a, int b) {
    for (unsigned int i = 0; i < a; i++) {
        for (unsigned int j = 0; j < b; j++) {
            delete[] v[i][j];
        }
        delete[] v[i];
    }
    delete[] v;
    v = nullptr;
}



#endif 
