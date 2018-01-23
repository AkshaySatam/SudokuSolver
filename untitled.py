
import cv2
import numpy as np
import joblib
import sys
import sudokusolver
"""
try, try2, try3, try4, try5, a ,try6, try9, try11, TRY9<<========= Working
try1, try10, try14, border <<============= pyrDown k saath chalta hai.
try12, try13 <===== resize
"""
# try5, try6, try8, try12, try13
# try13 301440
"""
border, try1, try11, try14.
"""





clf = joblib.load('classifier.pkl')
font = cv2.FONT_HERSHEY_SIMPLEX
# Loading image contains lines
img2 = cv2.imread(sys.argv[1])
final=[]
diffx=0
diffy=0
test = 3
while test != 4:

	img = cv2.imread(sys.argv[1])
	# # if img.size > 1000000 and test == 0:
	# img = cv2.resize(img, (346,336))
	if test ==1:
		img = cv2.resize(img, (346,336))
	elif test == 2:
		print "inelse"
		img = cv2.pyrDown(img)	
	elif test == 3:
		print "in here"
		img = cv2.pyrDown(img)	
		img = cv2.resize(img, (346,336))
	h,w,k = img.shape
	print img.size, h, w, test
	# Convert to grayscale
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Apply Canny edge detection, return will be a binary image
	edges = cv2.Canny(img,50,100,apertureSize = 3)
	# Apply Hough Line Transform, minimum lenght of line is 200 pixels
	lines = cv2.HoughLines(edges,1,np.pi/180,200)	
	# Print and draw line on the original image

	l = []
	l1=[]
	# print type(lines), "?????????	"
	if lines is not None:
		for a in lines:
			l.append([a[0][0],a[0][1]])

		# print sorted(l)
		# print l, len(l)
		# l1=[]
		for i in l:
			# print i
			rho = i[0]
			theta = i[1]
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			# line_detection = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

			# print x1,y1,x2,y2
			if abs(x1) != 1000:
					l1.append([x1,0,1])

			if abs(y1) != 1000:
					l1.append([0,y1,0])
				# cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
		# cv2.imshow("Line Detection",line_detection)
		# cv2.waitKey(0)
		# print l1
		# print len(l1)
		k = []
		l=l1
		# print sorted(l)
		for a in l:
			if a[2] == 1:
				for i in l:
					if i[2] == 0:
						k.append([a[0],i[1]])
		# print k, len(k)
		# print sorted(k), len(k)
		# kl=[]
		k= sorted(k)
		i=0;
		# print k, len(k)
		while i < len(k)-1:
			# print k[i],k[i+1],"????????",k[i+1][1] - k[i][1]
			if k[i+1][1] - k[i][1] <= 10 and k[i+1][1] - k[i][1]>0:
				k[i][1] = (k[i][1]+k[i+1][1])/2
				k.pop(i+1)
			# print k[i],"......."
			i=i+1

		i=0
		# print "\n\n\n\n",k, len(k)
		boom=[]
		flag=0
		corners = img.copy()
		if len(k) > 99:
			while i < len(k)-10:
				# print k[i][0]," ",k[i+10][0]
				if - k[i][0] + k[i+10][0] <= 10:
					k[i][0] = (k[i][0]+k[i+10][0])/2
					# corners = cv2.circle(corners, (k[i][0],k[i][1]), 5, (255,0,0), 3)
					boom.append(k[i])
					flag=0
				else:
					# print "in else",k[i][0]," ",k[i+10][0]
					boom.append(k[i])
					flag = 1
					# corners = cv2.circle(corners, (k[i][0],k[i][1]), 5, (255,0,0), 3)
				i=i+1
				if flag ==0:
					if i%10 == 0:
						i+=10
			
			# print len(boom),"::::::::::::::::::::"
			if len(boom) != 100:
				for i in range(len(k)-10, len(k)):
					# corners = cv2.circle(corners, (k[i][0],k[i][1]), 5, (255,0,0), 3)
					boom.append(k[i])
			# cv2.imshow("Intersection", corners)
			# cv2.waitKey(0)
			# print boom, len(boom)

			prev = boom[11][0]
			j=0
			k=0
			ans=[]
			ans1=[]
			p=[]
			p1=[]
			# print boom
			# print (boom[len(boom)-1][0]-boom[0][0])/10,(boom[len(boom)-1][1]-boom[0][1])/10 
			for i in range(len(boom)-11):
				if boom[i+11][0] == prev:
					x1 = boom[i][0]
					y1 = boom[i][1]
					x2 = boom[i+11][0]
					y2 = boom[i+11][1]
					# print x1,x2,x2-x1,y1,y2,y2-y1
					diff_x =int((x2-x1)/5)
					diff_y = int((y2-y1)/5)
					# diff_y1 = int((y2-y1)/4)
					x1 = x1 + diff_x
					x2 = x2 - diff_x
					y1 = y1 + diff_y
					y2 = y2 - diff_y
					# print x1,x2,x2-x1,y1,y2,y2-y1
					# cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
					X = img[y1:y2,x1:x2]
					if X.size !=0:
						X = cv2.bitwise_not(X)
						X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
						X = cv2.resize(X, (36,36))
						img1 = np.reshape(X,(1,-1))
						num = clf.predict(img1)
						# print "Predicted Number ======>  ", num
						# cv2.imshow("cell_predict", X)
						# cv2.waitKey(0)
						
						# if (num[0] != 0):
						# 	cv2.putText(img,str(num[0]),(boom[i][0]+13,boom[i][1]+25),font,0.75,(225,0,0),2)
						# else:
						# 	cv2.putText(img,str(num[0]),(boom[i][0]+17,boom[i][1]+30),font,1,(225,0,0),2)
						ans.append(num[0])
						p.append([x1,y1])
						k+=1
						if k == 9:
							k=0
							ans1.append(ans)
							ans=[]
							p1.append(p)
							p=[]
							
				else:
					prev = boom[i+11][0]
			if test == 0:
				final = p1
				diffx=diff_x
				diffy =diff_y
			flag = 0
			for i in ans1:
				for j in i:
					if list(i).count(j) > 1 and j != 0:
						flag =1
						break;
			if len(ans1)!=9 and len(ans1[0])!=9:
				flag = 1
			sudoku = np.array(ans1).transpose()
			if flag==0:
				for i in sudoku:
					for j in i:
						if list(i).count(j) > 1 and j != 0:
							flag = 1
							break;
			print "Input Sudoku\n", sudoku,"\n\n"
			grid=""
			for i in sudoku:
				for j in i:
					grid+=str(j)
			# print grid
			if(flag==0):

				sudoku = sudokusolver.run(grid)
				# print sudoku,".............\n\n\n\n", len(sudoku)
				if len(sudoku) == 9:
					print "Solved Sudoku\n",np.array(sudoku)
					sudoku= np.array(sudoku).transpose()
					for i in range(0,9):
						for j in range(0,9):
								if(ans1[i][j]==0):
									cv2.putText(img,str(sudoku[i][j]),(p1[i][j][0]+diff_x,p1[i][j][1]+3*diff_y),font,0.75,(0,255,0),2)
					cv2.imshow("Solved Sudoku", img)
					cv2.waitKey(0)
					break;
				else:
					print "Wrong data extracted!!!"
					test+=1
			else:
				print "Digits extracted are wrong!! ",
				test+=1
		else:
			print "Rectangles not found"
			test+=1
	else:
		print "Error in image"
		test+=1
# Show the result
cv2.destroyAllWindows()
