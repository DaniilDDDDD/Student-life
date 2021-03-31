def merge_sort(array, begin, end):
    print(f'{begin+1} {end+1} {array[0]} {array[len(array)-1]}')
    if len(array) == 1:
        return array
    

    left_end = begin + len(array)//2 - 1
    right_begin = left_end + 1   
    
    left = merge_sort(array[0 : len(array)//2], begin, left_end)

    right = merge_sort(array[len(array)//2 : len(array)], right_begin, end)
    result = [0] * len(array)
  

    l, r, k = 0, 0, 0
    while l < len(left) and r < len(right):
        if left[l] <= right[r]: 
            result[k] = left[l]
            l += 1
        else:
            result[k] = right[r]
            r += 1
        k += 1

    while l < len(left): 
        result[k] = left[l] 
        l += 1
        k += 1  
    
    while r < len(right): 
        result[k] = right[r]
        r += 1
        k += 1
  
    return result 


def main():
    n = int(input())
    array = [int(x) for x in input().split()]
    print(' '.join(map(str,merge_sort(array, 0, len(array)-1))))


if __name__=='__main__':
    main()