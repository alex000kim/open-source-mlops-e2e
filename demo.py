from random import randint

def ticket_number(name):
    print(f"Hi {name}, your ticket number is {randint(1, 100)}")

if __name__ == "__main__":
    ticket_number("Marco")