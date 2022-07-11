import cv2
import crop_img as crop
def front_info(img):
    number= img[97:97+40,310:310+260].copy()
    face=img[167:167+200,37:37+152].copy()
    name1=img[138:138+42,280:280+355].copy()
    name2=img[180:180+42,210:210+410].copy()
    birth_date=img[220:220+35,320:320+250].copy()
    birth_place1=img[255:255+35,338:338+295].copy()
    birth_place2=img[290:290+40,210:210+410].copy()
    hk1=img[325:325+40,407:407+292].copy()
    hk2=img[360:360+40,210:210+410].copy()
    return number,name1,name2,birth_date,birth_place1,birth_place2,hk1,hk2,face

def back_info(img):	
	left= img[62 : 62+170, 10: 10+215].copy()
	right= img[232 : 232+170,10: 10+215].copy()
	dantoc= img[15 : 15+50, 110: 110+200].copy()
	tongiao= img[15 : 15+50, 410: 410+220].copy()
	dauhieu= img[95 : 95+100, 257: 257+385].copy()
	return left, right, dantoc, tongiao, dauhieu

def back_id(img):	
	left= img[100 : 100+155, 0: 0+185].copy()
	right= img[260 : 260+155,0: 0+185].copy()
	dauhieu1= img[107 : 107+35, 370: 370+250].copy()
	dauhieu2= img[137 : 137+35, 215: 215+350].copy()
	where1= img[225 : 225+26, 345: 345+280].copy()
	where2= img[245 : 245+26, 300: 300+330].copy()
	return left, right, dauhieu1, dauhieu2

def front_id(img):	
    number= img[107:107+40,295:300+280].copy()
    face=img[140:140+235,21:21+180].copy()
    name=img[155:155+55,310:310+280].copy()
    birth_date=img[210:210+30,390:390+125].copy()
    gender=img[245:245+35,290:290+65].copy()
    quoc_tich=img[245:245+35,470:470+110].copy()
    que_quan1=img[283:283+40,300:300+330].copy()
    que_quan2=img[305:305+35,260:260+380].copy()
    ho_khau1=img[335:335+32,322:322+296].copy() #Hộ khẩu dòng 1
    ho_khau2=img[364:364+35,260:260+380].copy() #Hộ khẩu dòng 2
    valid=img[375:375+33,130:130+130].copy()
    return number, name, birth_date, gender, quoc_tich, que_quan1,que_quan2, ho_khau1, ho_khau2, valid, face

def choice_filter(image):
	filter1=crop.Front_Filter(image)
	if len(filter1)>300:
		# print("Filter 1")
		return filter1

	filter2=crop.Back_Filter(image)
	if len(filter2)>300:
		# print("Filter 2")
		return filter2

	filter3=crop.ID_Filter(image)
	if len(filter3)>300:
		# print("Filter 3")
		return filter3	

	filter4=crop.Extend_Filter(image)
	if len(filter4)>300:
		# print("Filter 4")
		return filter4
	return []

#Nhận dạng 2 mặt của 2 loại thẻ căn cước.
def choice_info(img):
	# Xác định con dấu đỏ trên mặt trước góc bên trái
	label1= img[10:10+130,50:50+145].copy()

	#Chuyển sang hệ màu HSV
	img_hsv = cv2.cvtColor(label1, cv2.COLOR_BGR2HSV)

	#Nhận biết 3 màu của của con dấu đỏ 
	# 1,2: đỏ hồng và 3: vàng -cam
	mask1 = cv2.inRange(img_hsv, (0,50,20), (10,255,255))
	mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
	mask3 = cv2.inRange(img_hsv, (10,100,100), (25,255,255))

	# gọp 3 màu của con dấu vào một
	mask = cv2.bitwise_or(mask1, mask2)
	mask = cv2.bitwise_or(mask, mask3)

	#Đếm số điểm màu xem có phải con dấu đỏ nếu đúng tiếp tục phân loại 2 mặt cmnd
	if cv2.countNonZero(mask) > 3500:
		#Nhận dạng dòng chữ dưới ảnh của thẻ căn cước
		label2=img[370:370+40,25:25+190].copy()
		label2 = cv2.inRange(cv2.cvtColor(label2, cv2.COLOR_BGR2HSV), (0, 0, 0),(255, 150, 110))
		# cv2.imshow("label2",label2)
		# cv2.waitKey(0)
		if cv2.countNonZero(label2) > 120:  #Đếm số điểm ảnh để xác định chữ dưới ảnh
			return 2 #Mat truoc can cuoc(Dựa vào số dưới ảnh)
		else:
			return 1 #Đây là mặt trước CMND
	else:
		#Nhận dạng mặt sau, chuyển qua nhận dạng xem có mã barcode phía trên cùng không bằng cách đếm số điểm ảnh
		label3= img[30:30+60,50:50+590].copy()
		gray_label3 = cv2.cvtColor(label3,cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray_label3, 130, 250, cv2.THRESH_BINARY)[1]
		#đếm số điểm ảnh trên label 3 xem có phải bar code không 
		# nếu có là mặt sau căn cước
		if(cv2.countNonZero(thresh)) >=25000: 
			return 3 # mat sau CMND
		else: 
			return 4 # mat sau can cuoc

#Dùng để cắt các thông tin trong CMND và căn cước theo loại
def split_image(image):
	warped = choice_filter(image)
	if len(warped)>0:
		# Resize về dạng chuẩn truóc khi xử lý
		warped = cv2.resize(warped,(648,408))
		typed= choice_info(warped)
		# print("loai",typed)
		if typed==1:
			print("Mặt trước Chứng minh nhân dân cũ")
			number,name1,name2,birth_date,birth_place1,birth_place2,hk1,hk2,face=front_info(warped)
			return number,name1,name2,birth_date,birth_place1,birth_place2,hk1,hk2,face,warped,typed
			#number: So CMND
			#name1: Ho va ten dong 1
			#name2: Ho va ten dong 2
			#birth_date: Ngay sinh
			#birth_place1: nguyen quan dong 1
			#birth_place2: nguyen quan dong 2
			#come_from: que quan( noi sinh)
			#nationality: quoc tich
			#Hk1: ho khau dong 1
			#HK1: Ho khau dong 2
			#face: Anh chan dung
			#warped: Anh cat cmmnd 
			#type: mac dinh la 1

		elif typed ==2:
			print("Mặt trước căn cước công dân")
			number, name, birth_date, gender, quoc_tich, que_quan1,que_quan2, ho_khau1, ho_khau2, valid, face=front_id(warped)
			return [number, name, birth_date, gender, quoc_tich, que_quan1,que_quan2, ho_khau1, ho_khau2, valid, face,warped,typed]
			#number: So CMND
			#name: Ho va ten
			#birth_date: Ngay sinh
			#gender: gioi tinh
			#que_quan(1 va 2): que quan( noi sinh)
			#quoc tich: quoc tich
			#hokhau1: ho khau dong 1
			#hokhau2: Ho khau dong 2
			#Valid: thoi han su dung( duoi anh chan dung)
			#face: Anh chan dung
			#warped: Anh cat cmmnd 
			#type: mac dinh la 2

		elif typed ==3:
			print("Mặt sau chứng minh nhân dân")
			left, right, dantoc, tongiao, dauhieu=back_info(warped)
			return dantoc, tongiao, dauhieu,left,right,warped,typed
			#left: van tay trai
			#right: van tay phai
			#dantoc:  Ten dan toc
			#tongiao: ten ton giao
			#warped: Anh cat cmmnd 
			#type: mac dinh la 3

		elif typed ==4:
			print("Mặt sau căn cước công dân")
			left, right, dauhieu1, dauhieu2=back_id(warped)
			return [left, right, dauhieu1, dauhieu2,warped,typed]
			#left: van tay trai
			#right: van tay phai
			#dauhieu1:  Dau hien nhan dang dong 1
			#dauhieu2: Dau hien nhan dang dong 2
			#warped: Anh cat cmmnd 
			#type: mac dinh la 4