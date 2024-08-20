import random as rd

tebakan = rd.randint(0, 100) 
rangeawal = 0
rangeakhir = 100

print(tebakan)

while True:
    jawaban = input("Input: ")
    if jawaban == "=":
        print("Tebakan komputer benar!")
        break
    if jawaban == ">":
        rangeakhir = tebakan - 1
        tebakan = rd.randint(rangeawal, rangeakhir)
        print(tebakan)
    if jawaban == "<":
        rangeawal = tebakan + 1
        tebakan = rd.randint(rangeawal, rangeakhir)
        print(tebakan)
