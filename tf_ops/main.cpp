#include <iostream>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "Eigen/Dense"


int main() {
    int a[4]{0, 0, 0, 0};
    for (int i = 0; i < 4; i++){
        std::cout << a[i];
    }
    Eigen::TensorMap<Eigen::Tensor<int, 1>> t_map(a, 4);
    t_map.setConstant(2);
    for (int i = 0; i < 4; i++){
        std::cout << a[i];
    }
//    Eigen::Tensor<int, 2> tensor(2, 2);
//    Eigen::Tensor<int, 2> a(2, 3);
//    a.setValues({{1, 2, 3}, {6, 5, 3}});
//    Eigen::Tensor<int, 2> b(3, 2);
//    b.setValues({{1, 2}, {4, 5}, {5, 6}});
//    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
//    Eigen::Tensor<int, 2> AB = a.contract(b, product_dims);
//    std::cout << AB << std::endl;
//    Eigen::array<Eigen::IndexPair<int>, 1> transposed_product_dims = {Eigen::IndexPair<int>(0, 1)};
//    Eigen::Tensor<int, 2> AtBt = a.contract(b, transposed_product_dims);
//    std::cout << AtBt << std::endl;
//    Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(0, 1), Eigen::IndexPair<int>(1, 0) };
//    Eigen::Tensor<int, 0> AdoubleContractedA = a.contract(a, double_contraction_product_dims);
//    std::cout << AdoubleContractedA << std::endl;
//    Eigen::DefaultDevice dev;
//    int n = 2000;
//    Eigen::MatrixXd a = Eigen::MatrixXd::Ones(n, n);
//    Eigen::MatrixXd b = Eigen::MatrixXd::Ones(n, n);
//    Eigen::MatrixXd c;
//    Eigen::Tensor<int, 3> tensor(1, 2, 3);
//    Eigen::Tensor<int, 3> tensor1(1, 2, 3);
//    Eigen::Tensor<int, 3> tensor2(1, 2, 3);
//    tensor1.setConstant(1);
//    tensor2.setConstant(2);
//    tensor.device(dev) = tensor1 + tensor2;
//    std::cout << tensor << std::endl;
//    clock_t start = clock();
//    int N = 10;
//    for (int i = 0; i < N; i++){
//        c = a * b;
//    }
//    clock_t end = clock();
//    double elpased_time = (double(end) - double(start)) / CLOCKS_PER_SEC;
//    std::cout << "elapsed time is " << elpased_time << std::endl;
    return 0;
}