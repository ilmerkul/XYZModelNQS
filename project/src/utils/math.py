from typing import List


def powers_of_two(n: int) -> List[int]:
    powers = []
    power = 1

    while n > 0:
        if n % 2 == 1:
            powers.append(power)
        n //= 2
        power *= 2

    return powers
