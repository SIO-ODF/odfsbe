def get_volt_indicies(n):
    start = (n // 2 * 3)
    high = n % 2
    return start + high, start + high + 1, 1 - high

def get_voltage(hex, channel, freq_supressed):
    offset = (5 - freq_supressed) * 3
    
    first_byte_idx, second_byte_idx, shift = get_volt_indicies(channel)
    first_byte_idx += offset
    second_byte_idx += offset
    
    data = hex[:, first_byte_idx].astype("uint16") << 8
    data = data | hex[:, second_byte_idx]
    data = data >> (4 * shift)
    data = data & 4095
    return 5 * (1 - (data/4095))
    
def get_frequency(hex, channel):
    m = 3 * channel
    data = hex[:,m].astype("uint32") << 8
    data = (data | hex[:, m+1]) << 8
    data = data | hex[:, m+2]
    return data / 256