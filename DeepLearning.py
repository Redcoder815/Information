def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j]<= pivot:
            i = i + 1
            array[i], array[j] = array[j], array[i]
    array[i+1], array[high] = array[high], array[i+1]
    return i + 1

def quickSort(array, low, high):
    if low < high:
        partiton_point = partition(array, low, high)

        quickSort(array, low, partiton_point - 1)
        quickSort(array, partiton_point + 1, high)

array = [9, 5, 1, 4]
size = len(array)
quickSort(array, 0, size - 1)
print(array)