/*  C Program to Check whether a given Number is  Perfect Number */
#include <stdio.h>
 
int main()
{
    int n, rem, s = 0, i;
    scanf("%d", &n);
    for (i = 1; i <= (n - 1); i++)
    {
        rem = n % i;
	if (rem == 0)
        {
            s = s + i;
        }
    }
    if (s == n)
        printf("Entered Number is correct");
    else
        printf("Entered Number is not a perfect ");
    return 0;
}