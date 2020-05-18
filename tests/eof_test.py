try:
    while True:
        data = raw_input('prompt:')
        print('READ:', data)
    
except EOFError as e:
    print(e)

# $ echo hello | python eof_test.py