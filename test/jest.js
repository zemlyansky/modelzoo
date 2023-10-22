/**
 * Simple polyfill for Jest (faster debugging). For example:
 * node -r ./test/jest.js ./test/gpt_save.js
 */
global.test = (name, f) => {
    console.log('Test:', name)
    f()
}
global.expect = (x) => {
    return {
        toBe: (y) => {
            if (x != y) {
                console.error('Error:', x, '!=', y)
            }
        },
        toEqual: (y) => {
            if (x != y) {
                console.error('Error:', x, '!=', y)
            }
        },
        toBeLessThan: (y) => {
            if (x >= y) {
                console.error('Error:', x, '>=', y)
            }
        },
        toBeCloseTo: (y, n) => {
            if (Math.abs(x - y) >= n) {
                console.error('Error:', x, '!=', y)
            }
        },
        toBeDefined: () => {
            if (typeof x == 'undefined') {
                console.error('Error:', x, 'is undefined')
            }
        }
    }
}