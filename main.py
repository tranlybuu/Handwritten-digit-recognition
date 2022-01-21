from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RandInitialise import initialise
from Prediction import predict
from scipy.optimize import minimize
 
 
# Đọc dữ liệu từ file MNIST-ORIGINAL
data = loadmat('mnist-original.mat')
X = data['data']

# Chuẩn hóa dữ liệu
X = X.transpose()
X = X / 255
 
# Trích xuất nhãn từ tệp mat
y = data['label']
y = y.flatten()
 
# Tách dữ liệu thành tập huấn luyện với 60.000 mẫu
X_train = X[:60000, :]
y_train = y[:60000]
 
# Tách dữ liệu thành bộ thử nghiệm với 10.000 mẫu
X_test = X[10000:, :]
y_test = y[10000:]
 
m = X.shape[0]
input_layer_size = 784  # Hình ảnh có kích thước 28x28 nên sẽ chuyển thành 1x784
hidden_layer_size = 100
num_labels = 10  # Có 10 lớp là từ 0->9
 
# Khởi tạo ngẫu nhiên Thetas
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)
 
# Bỏ cuộn các tham số vào một vectơ cột duy nhất
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
maxiter = 100
lambda_reg = 0.1  # hạn chế overfitting
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)
 
# Gọi hàm minimize để giảm thiểu cost function
results = minimize(neural_network, x0=initial_nn_params, args=myargs,
          options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)
 
nn_params = results["x"]  # Theta được đào tạo được trích xuất
 
# Trọng số được chia trở lại Theta1, Theta2
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (
                              hidden_layer_size, input_layer_size + 1))  # shape = (100, 785)
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                      (num_labels, hidden_layer_size + 1))  # shape = (10, 101)
 
# Kiểm tra độ chính xác của bộ dữ liệu test model
pred = predict(Theta1, Theta2, X_test)
print('Độ chính xác của tập dữ liệu dùng để test: {:f}%'.format((np.mean(pred == y_test) * 100)))
 
# Kiểm tra độ chính xác của bộ dữ liệu train model
pred = predict(Theta1, Theta2, X_train)
print('Độ chính xác của tập dữ liệu dùng để train: {:f}%'.format((np.mean(pred == y_train) * 100)))
 
# Đánh giá độ chính xác của model
true_positive = 0
for i in range(len(pred)):
    if pred[i] == y_train[i]:
        true_positive += 1
false_positive = len(y_train) - true_positive
print(f'Độ chính xác = {true_positive/(true_positive + false_positive)}')
 
# Lưu Thetas trong file .txt
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')