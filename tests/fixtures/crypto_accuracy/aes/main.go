package main

import "crypto/aes"

func main() {
	_, _ = aes.NewCipher(make([]byte, 16))
}
