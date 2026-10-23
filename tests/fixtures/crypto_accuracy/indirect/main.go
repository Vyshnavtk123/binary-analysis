package main

import (
	"crypto/aes"
	"crypto/cipher"
)

type cipherFactory func([]byte) (cipher.Block, error)

var factory cipherFactory = aes.NewCipher

func main() {
	_, _ = factory(make([]byte, 16))
}
