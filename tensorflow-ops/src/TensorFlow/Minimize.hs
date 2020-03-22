-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

module TensorFlow.Minimize
    ( Minimizer
    , MinimizerRefs
    , minimizeWith
    , minimizeWithRefs
    , gradientDescent
    , gradientDescentRefs
    , AdamConfig(..)
    , adam
    , adamRefs
    , adam'
    , adamRefs'
    ) where

import           Control.Monad          (void, zipWithM)
import           Data.Default           (Default (..))
import           Data.Int               (Int64)
import           Data.List              (zipWith4, zipWith5)
import           Data.Maybe             (fromMaybe)


import qualified TensorFlow.Build       as TF (Build, explicitName)
import qualified TensorFlow.Core        as TF
import qualified TensorFlow.GenOps.Core as TFO (applyAdam, applyRMSProp, assignAdd,
                                                resourceApplyAdam, resourceApplyRMSProp,
                                                shape)
import qualified TensorFlow.Gradient    as TF
import qualified TensorFlow.Ops         as TF hiding (assign, initializedVariable)
import qualified TensorFlow.Ops         as TFO (assign, initializedVariable,
                                                initializedVariable', scalar,
                                                zeroInitializedVariable, zeros)
import qualified TensorFlow.Output      as TF (OpDef (..))
import qualified TensorFlow.Tensor      as TF (Tensor (..), TensorKind, renderValue,
                                               toBuild)
import qualified TensorFlow.Variable    as TF

-- | Functions that minimize a loss w.r.t. a set of 'TF.Variable's.
--
-- Generally only performs one step of an iterative algorithm.
--
-- 'Minimizer's are defined as a function of the gradients instead of
-- the loss so that users can apply transformations to the gradients.
type Minimizer a =
    forall m. TF.MonadBuild m =>
    [TF.Variable a] -> [TF.Tensor TF.Value a] -> m TF.ControlNode

type MinimizerRefs a =
    forall m. TF.MonadBuild m =>
    [TF.Tensor TF.Ref a] -> [TF.Shape] -> [TF.Tensor TF.Value a] -> m (TF.ControlNode, [TF.Tensor TF.Ref a], [TF.Tensor TF.Ref a])


-- | Convenience wrapper around 'TF.gradients' and a 'Minimizer'.
minimizeWith :: (TF.MonadBuild m, TF.GradientCompatible a)
             => Minimizer a
             -> TF.Tensor v a    -- ^ Loss.
             -> [TF.Variable a]  -- ^ Parameters of the loss function.
             -> m TF.ControlNode
minimizeWith minimizer loss params =
    TF.gradients loss params >>= minimizer params


-- | Convenience wrapper around 'TF.gradients' and a 'Minimizer'.
minimizeWithRefs :: (TF.MonadBuild m, TF.GradientCompatible a)
             => MinimizerRefs a
             -> TF.Tensor v a        -- ^ Loss.
             -> [TF.Tensor TF.Ref a] -- ^ Parameters of the loss function.
             -> [TF.Shape]           -- ^ Parameters shapes of the loss function.
             -> m (TF.ControlNode, [TF.Tensor TF.Ref a], [TF.Tensor TF.Ref a])
minimizeWithRefs minimizer loss params shapes = do
    let vals = map TF.value params
    TF.gradients loss vals >>= minimizer params shapes


-- | Perform one step of the gradient descent algorithm.
gradientDescent :: TF.GradientCompatible a
                => a  -- ^ Learning rate.
                -> Minimizer a
gradientDescent learningRate params grads = TF.withNameScope "gradientDescent" $ do
    let applyGrad param grad =
            TF.assignAdd param (TF.scalar (-learningRate) `TF.mul` grad)
    TF.group =<< zipWithM applyGrad params grads

-- | Perform one step of the gradient descent algorithm.
gradientDescentRefs :: TF.GradientCompatible a
                => a  -- ^ Learning rate.
                -> MinimizerRefs a
gradientDescentRefs learningRate params _ grads =
  TF.withNameScope "gradientDescent" $ do
    lrRef <- TFO.initializedVariable' (\x -> x {TF._opName = TF.explicitName "learningRate"}) (TF.scalar learningRate)
    let applyGrad param grad = TFO.assignAdd param (TF.neg lrRef `TF.mul` grad)
    gr <- TF.group =<< zipWithM applyGrad params grads
    return (gr, [], [lrRef])

-- TODO: Support more than Float in adam.

data AdamConfig = AdamConfig
    { adamLearningRate :: Float -- ^ Learning rate [Default: 0.001]
    , adamBeta1        :: Float -- ^ Beta 1 [Default: 0.9]
    , adamBeta2        :: Float -- ^ Beta 2 [Default 0.999]
    , adamEpsilon      :: Float -- ^ Epsilon [Default: 1e-8]
    }

data RMSPropConfig = RMSPropConfig
    { rmsPropLearningRate :: Float -- ^ Learning rate [Default: 0.001]
    , rmsPropRho          :: Float -- ^ Decay rate [Default: 0.9]
    , rmsPropMomentum     :: Float -- ^ Momentum [Default 0.0]
    , rmsPropEpsilon      :: Float -- ^ Ridge Term [Default: 1e-7]
    }


instance Default AdamConfig where
  -- Recommended defaults from the adam paper.
  def = AdamConfig 0.001 0.9 0.999 1e-8

instance Default RMSPropConfig where
  def = RMSPropConfig 0.001 0.9 0.0 1e-7

-- | Perform one step of the adam algorithm.
--
-- See https://arxiv.org/abs/1412.6980.
--
-- NOTE: Currently requires all 'TF.Variable's to have an 'TF.initializedValue'.
adam :: Minimizer Float
adam = adam' def

adamRefs :: MinimizerRefs Float
adamRefs = adamRefs' def

rmsPropRefs :: MinimizerRefs Float
rmsPropRefs = rmsPropRefs' def


adam' :: AdamConfig -> Minimizer Float
adam' config params grads =
  TF.withNameScope "adam" $ do
    let lr = TF.scalar (adamLearningRate config)
        beta1 = TF.scalar (adamBeta1 config)
        beta2 = TF.scalar (adamBeta2 config)
        epsilon = TF.scalar (adamEpsilon config)
    -- Create adam state variables.
    let errorMsg = "TensorFlow.Minimize.adam requires an initial value for all tensors/variables"
        initVal = fromMaybe (error errorMsg) . TF.initializedValue
    ms <- mapM (TF.initializedVariable . TF.zerosLike . initVal) params
    vs <- mapM (TF.initializedVariable . TF.zerosLike . initVal) params
    beta1Power <- TF.initializedVariable beta1
    beta2Power <- TF.initializedVariable beta2
    -- Perform adam update.
    let applyGrad param m v = TF.resourceApplyAdam param m v (TF.readValue beta1Power) (TF.readValue beta2Power) lr beta1 beta2 epsilon
    updateVars <- sequence $ zipWith4 applyGrad params ms vs grads
    -- Update beta variables after adam update.
    let updateBeta betaPower beta = TF.withControlDependencies updateVars (TF.assign betaPower (TF.readValue betaPower `TF.mul` beta))
    updateBeta1 <- updateBeta beta1Power beta1
    updateBeta2 <- updateBeta beta2Power beta2
    TF.group (updateBeta1 : updateBeta2 : updateVars)

toBuildTensor :: TF.TensorKind v => TF.Tensor v a -> TF.Tensor TF.Build a
toBuildTensor (TF.Tensor o) = TF.Tensor $ TF.toBuild o

adamRefs' :: AdamConfig -> MinimizerRefs Float
adamRefs' config params shapes grads =
  TF.withNameScope "adamRefs" $ do
    lrRef <- TFO.initializedVariable' (\x -> x {TF._opName = TF.explicitName "learningRate"}) (TF.scalar $ adamLearningRate config)
    let lr = toBuildTensor lrRef
    let beta1 = TF.scalar (adamBeta1 config)
        beta2 = TF.scalar (adamBeta2 config)
        epsilon = TF.scalar (adamEpsilon config)
    -- Create adam state variables.
    ms <- mapM TFO.zeroInitializedVariable shapes
    vs <- mapM TFO.zeroInitializedVariable shapes
    beta1Power <- TFO.initializedVariable beta1
    beta2Power <- TFO.initializedVariable beta2
    -- Perform adam update.
    let applyGrad param m v = TFO.applyAdam param m v beta1Power beta2Power lr beta1 beta2 epsilon
    -- let applyGrad param m v = TF.resourceApplyAdam param m v (TF.readValue beta1Power) (TF.readValue beta2Power) lr beta1 beta2 epsilon

    updateVars <- sequence $ zipWith4 applyGrad params ms vs grads
    -- Update beta variables after adam update.
    let updateBeta betaPower beta = TF.withControlDependencies updateVars (TFO.assign betaPower (betaPower `TF.mul` beta))
    updateBeta1 <- updateBeta beta1Power beta1
    updateBeta2 <- updateBeta beta2Power beta2
    grp <- TF.group (updateBeta1 : updateBeta2 : updateVars)
    let vars = ms ++ vs ++ [beta1Power, beta2Power]
    return (grp, vars, [lrRef])

rmsPropRefs' :: RMSPropConfig -> MinimizerRefs Float
rmsPropRefs' config params shapes grads =
  TF.withNameScope "rmsPropRefs" $ do
    lrRef <- TFO.initializedVariable' (\x -> x {TF._opName = TF.explicitName "learningRate"}) (TF.scalar $ rmsPropLearningRate config)
    let lr = toBuildTensor lrRef
    let rho = TF.scalar (rmsPropRho config)
        momentum = TF.scalar (rmsPropMomentum config)
        epsilon = TF.scalar (rmsPropEpsilon config)
    -- Create rmsProp state variables.
    vs <- mapM TFO.zeroInitializedVariable shapes
    ms <- mapM TFO.zeroInitializedVariable shapes
    moms <- mapM TFO.zeroInitializedVariable shapes
    -- -- Perform rmsProp update.
    let applyGrad param v m mom = TFO.applyRMSProp v m mom lr rho momentum epsilon
    void $ sequence $ zipWith5 applyGrad params vs ms moms grads
    -- Update beta variables after rmsProp update.
    -- let updateBeta betaPower beta = TF.withControlDependencies updateVars (TFO.assign betaPower (betaPower `TF.mul` beta))
    -- updateBeta1 <- updateBeta beta1Power beta1
    -- updateBeta2 <- updateBeta beta2Power beta2
    -- grp <- TF.group updateVars
    grp <- TF.noOp
    let vars = ms ++ vs  ++ moms -- ++ [beta1Power, beta2Power]
    return (grp, vars, [lrRef])
