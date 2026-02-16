def sum(n):
    
    # base condition
    if n == 1:
        return 1
    
    return n + sum(n - 1)

if __name__ == "__main__":
    n = 5
    print(sum(n))