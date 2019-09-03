def bin_search(a, target):
    low, high = 0, len(a)
    while low < high:
        mid = (low + high) // 2
        if a[mid] < target:
            low = mid + 1
        else:
            high = mid
    return mid


bin_search([i for i in range(100)], 6)
