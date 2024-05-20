def sayHello(name):
      return f"Hello, {name}!"


def greet():
    name = input("Enter your name: ")
    greeting = sayHello(name)
    print(greeting)


greet()