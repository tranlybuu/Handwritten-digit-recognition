from tkinter import *
import numpy as np
from PIL import ImageGrab
from Prediction import predict

window = Tk()
window.title("Nhận dạng chữ viết tay")
l1 = Label()


def MyProject():
	global l1

	widget = cv
	# Đặt tọa độ canvas
	x = window.winfo_rootx() + widget.winfo_x()
	y = window.winfo_rooty() + widget.winfo_y()
	x1 = x + widget.winfo_width()
	y1 = y + widget.winfo_height()

	# Hình ảnh được chụp từ canvas và được thay đổi kích thước thành 28 X 28 pixel
	img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))

	# Chuyển ảnh có bảng màu RGB sang ảnh xám
	img = img.convert('L')

	# Trích xuất ma trận pixel của hình ảnh và chuyển đổi nó thành vectơ của 1x784
	x = np.asarray(img)
	vec = np.zeros((1, 784))
	k = 0
	for i in range(28):
		for j in range(28):
			vec[0][k] = x[i][j]
			k += 1

	# Tải dữ liệu từ 2 tập tin Theta
	Theta1 = np.loadtxt('Theta1.txt')
	Theta2 = np.loadtxt('Theta2.txt')

	# Gọi hàm dự đoán
	pred = predict(Theta1, Theta2, vec / 255)

	# Hiển thị kết quả
	l1 = Label(window, text="Dự đoán đây là số " + str(pred[0]), font=('Arial', 20))
	l1.place(x=170, y=420)


lastx, lasty = None, None


# Xóa canvas
def clear_widget():
	global cv, l1
	cv.delete("all")
	l1.destroy()


# Kích hoạt canvas
def event_activation(event):
	global lastx, lasty
	cv.bind('<B1-Motion>', draw_lines)
	lastx, lasty = event.x, event.y


# Để vẽ trên canvas
def draw_lines(event):
	global lastx, lasty
	x, y = event.x, event.y
	cv.create_line((lastx, lasty, x, y), width=15, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
	lastx, lasty = x, y


# Tên GUI
L1 = Label(window, text="Nhận diện chữ viết tay", font=('Arial', 25), fg="blue")
L1.place(x=130, y=10)

# Nút để xóa canvas
b1 = Button(window, text="Xóa", width=7, font=('Arial', 15), bg="orange", fg="black", command=clear_widget)
b1.place(x=120, y=370)

# Nút dự đoán chữ số được vẽ trên canvas
b2 = Button(window, text="Dự đoán", font=('Arial', 15), bg="yellow", fg="black", command=MyProject)
b2.place(x=380, y=370)

# Đặt kích thước của canvas
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)

cv.bind('<Button-1>', event_activation)
window.geometry("600x500")
window.mainloop()
