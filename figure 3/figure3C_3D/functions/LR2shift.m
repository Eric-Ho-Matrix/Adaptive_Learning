
% This is a code for finding the corresponding context shift for different 
% learning rates:

function [s] = LR2shift(allContextIncs, LRs, val)

% Inputs:
% allContextIncs = vector of all context shift increments
% LRs            = vector of all learning rates
% val            =   desired learning rate

% Output:        
% s              = corresponding context shift for "val"

% Find the coefficients.
coeffs = polyfit(allContextIncs,LRs, 10);

interpolatedX = linspace(0, 2, 500);
interpolatedY = polyval(coeffs, interpolatedX);

% Closest value in interpolatedY to desired learning rate:
[~, ix ] = min( abs( interpolatedY-val ) );
 
 % Corresponding Context shift:
 s=interpolatedX(ix);