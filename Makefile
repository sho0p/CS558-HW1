all:
	g++ hw1.cpp -o hw1.bin -Wall -g -Wextra -lpng

clean:
	rm hw1.bin
