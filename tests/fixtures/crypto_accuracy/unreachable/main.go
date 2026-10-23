package main

import "crypto/aes"

// deadCrypto is intentionally not referenced from main.
func deadCrypto() {
	_, _ = aes.NewCipher(make([]byte, 16))
}

func main() {}
