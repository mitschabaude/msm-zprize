(module
  (type (;0;) (func (param i32 i32)))
  (type (;1;) (func (param i32 i32 i32)))
  (type (;2;) (func (param i32)))
  (type (;3;) (func))
  (func (;0;) (type 0) (param i32 i32)
    (local i32)
    i32.const 0
    local.set 2
    loop  ;; label = @1
      local.get 0
      local.get 0
      local.get 0
      call 1
      local.get 2
      i32.const 1
      i32.add
      local.tee 2
      local.get 1
      i32.ne
      br_if 0 (;@1;)
    end)
  (func (;1;) (type 1) (param i32 i32 i32)
    (local i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i32)
    local.get 2
    i32.load
    i64.extend_i32_u
    local.set 6
    local.get 2
    i32.load offset=4
    i64.extend_i32_u
    local.set 7
    local.get 2
    i32.load offset=8
    i64.extend_i32_u
    local.set 8
    local.get 2
    i32.load offset=12
    i64.extend_i32_u
    local.set 9
    local.get 2
    i32.load offset=16
    i64.extend_i32_u
    local.set 10
    local.get 2
    i32.load offset=20
    i64.extend_i32_u
    local.set 11
    local.get 2
    i32.load offset=24
    i64.extend_i32_u
    local.set 12
    local.get 2
    i32.load offset=28
    i64.extend_i32_u
    local.set 13
    local.get 2
    i32.load offset=32
    i64.extend_i32_u
    local.set 14
    i32.const 0
    local.set 24
    loop  ;; label = @1
      local.get 1
      local.get 24
      i32.add
      i32.load
      i64.extend_i32_u
      local.set 5
      local.get 5
      local.get 6
      i64.mul
      local.get 15
      i64.add
      local.set 3
      i64.const 536870912
      local.get 3
      i64.const 536870911
      i64.and
      i64.sub
      local.set 4
      local.get 3
      local.get 4
      i64.add
      i64.const 29
      i64.shr_u
      local.get 16
      i64.add
      local.get 5
      local.get 7
      i64.mul
      i64.add
      local.get 4
      i64.const 157910888
      i64.mul
      i64.add
      local.set 15
      local.get 17
      local.get 5
      local.get 8
      i64.mul
      i64.add
      local.get 4
      i64.const 322848486
      i64.mul
      i64.add
      local.set 16
      local.get 18
      local.get 5
      local.get 9
      i64.mul
      i64.add
      local.get 4
      i64.const 221378578
      i64.mul
      i64.add
      local.set 17
      local.get 19
      local.get 5
      local.get 10
      i64.mul
      i64.add
      local.get 4
      i64.const 548
      i64.mul
      i64.add
      local.set 18
      local.get 20
      local.get 5
      local.get 11
      i64.mul
      i64.add
      local.set 19
      local.get 21
      local.get 5
      local.get 12
      i64.mul
      i64.add
      local.set 20
      local.get 22
      local.get 5
      local.get 13
      i64.mul
      i64.add
      local.set 21
      local.get 5
      local.get 14
      i64.mul
      local.get 4
      i64.const 4194304
      i64.mul
      i64.add
      local.set 22
      local.get 24
      i32.const 4
      i32.add
      local.tee 24
      i32.const 36
      i32.ne
      br_if 0 (;@1;)
    end
    local.get 0
    local.get 15
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store
    local.get 15
    i64.const 29
    i64.shr_u
    local.get 16
    i64.add
    local.set 16
    local.get 0
    local.get 16
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=4
    local.get 16
    i64.const 29
    i64.shr_u
    local.get 17
    i64.add
    local.set 17
    local.get 0
    local.get 17
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=8
    local.get 17
    i64.const 29
    i64.shr_u
    local.get 18
    i64.add
    local.set 18
    local.get 0
    local.get 18
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=12
    local.get 18
    i64.const 29
    i64.shr_u
    local.get 19
    i64.add
    local.set 19
    local.get 0
    local.get 19
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=16
    local.get 19
    i64.const 29
    i64.shr_u
    local.get 20
    i64.add
    local.set 20
    local.get 0
    local.get 20
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=20
    local.get 20
    i64.const 29
    i64.shr_u
    local.get 21
    i64.add
    local.set 21
    local.get 0
    local.get 21
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=24
    local.get 21
    i64.const 29
    i64.shr_u
    local.get 22
    i64.add
    local.set 22
    local.get 0
    local.get 22
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=28
    local.get 22
    i64.const 29
    i64.shr_u
    local.get 23
    i64.add
    local.set 23
    local.get 0
    local.get 23
    i32.wrap_i64
    i32.store offset=32)
  (func (;2;) (type 0) (param i32 i32)
    (local i32)
    i32.const 0
    local.set 2
    loop  ;; label = @1
      local.get 0
      local.get 0
      local.get 0
      call 3
      local.get 2
      i32.const 1
      i32.add
      local.tee 2
      local.get 1
      i32.ne
      br_if 0 (;@1;)
    end)
  (func (;3;) (type 1) (param i32 i32 i32)
    (local i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i32)
    local.get 2
    i32.load
    i64.extend_i32_u
    local.set 5
    local.get 2
    i32.load offset=4
    i64.extend_i32_u
    local.set 6
    local.get 2
    i32.load offset=8
    i64.extend_i32_u
    local.set 7
    local.get 2
    i32.load offset=12
    i64.extend_i32_u
    local.set 8
    local.get 2
    i32.load offset=16
    i64.extend_i32_u
    local.set 9
    local.get 2
    i32.load offset=20
    i64.extend_i32_u
    local.set 10
    local.get 2
    i32.load offset=24
    i64.extend_i32_u
    local.set 11
    local.get 2
    i32.load offset=28
    i64.extend_i32_u
    local.set 12
    local.get 2
    i32.load offset=32
    i64.extend_i32_u
    local.set 13
    i32.const 0
    local.set 23
    loop  ;; label = @1
      local.get 1
      local.get 23
      i32.add
      i32.load
      i64.extend_i32_u
      local.set 4
      local.get 14
      local.get 4
      local.get 5
      i64.mul
      i64.add
      local.set 3
      local.get 0
      local.get 23
      i32.add
      local.get 3
      i64.const 536870911
      i64.and
      i32.wrap_i64
      i32.store
      local.get 3
      i64.const 29
      i64.shr_u
      local.get 15
      i64.add
      local.get 4
      local.get 6
      i64.mul
      i64.add
      local.set 14
      local.get 16
      local.get 4
      local.get 7
      i64.mul
      i64.add
      local.set 15
      local.get 17
      local.get 4
      local.get 8
      i64.mul
      i64.add
      local.set 16
      local.get 18
      local.get 4
      local.get 9
      i64.mul
      i64.add
      local.set 17
      local.get 19
      local.get 4
      local.get 10
      i64.mul
      i64.add
      local.set 18
      local.get 20
      local.get 4
      local.get 11
      i64.mul
      i64.add
      local.set 19
      local.get 21
      local.get 4
      local.get 12
      i64.mul
      i64.add
      local.set 20
      local.get 23
      i32.const 4
      i32.add
      local.tee 23
      i32.const 36
      i32.ne
      br_if 0 (;@1;)
    end
    local.get 14
    local.set 3
    local.get 0
    local.get 3
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=36
    local.get 3
    i64.const 29
    i64.shr_u
    local.get 15
    i64.add
    local.set 15
    local.get 15
    local.set 3
    local.get 0
    local.get 3
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=40
    local.get 3
    i64.const 29
    i64.shr_u
    local.get 16
    i64.add
    local.set 16
    local.get 16
    local.set 3
    local.get 0
    local.get 3
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=44
    local.get 3
    i64.const 29
    i64.shr_u
    local.get 17
    i64.add
    local.set 17
    local.get 17
    local.set 3
    local.get 0
    local.get 3
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=48
    local.get 3
    i64.const 29
    i64.shr_u
    local.get 18
    i64.add
    local.set 18
    local.get 18
    local.set 3
    local.get 0
    local.get 3
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=52
    local.get 3
    i64.const 29
    i64.shr_u
    local.get 19
    i64.add
    local.set 19
    local.get 19
    local.set 3
    local.get 0
    local.get 3
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=56
    local.get 3
    i64.const 29
    i64.shr_u
    local.get 20
    i64.add
    local.set 20
    local.get 20
    local.set 3
    local.get 0
    local.get 3
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=60
    local.get 3
    i64.const 29
    i64.shr_u
    local.get 21
    i64.add
    local.set 21
    local.get 21
    local.set 3
    local.get 0
    local.get 3
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=64
    local.get 3
    i64.const 29
    i64.shr_u
    local.get 22
    i64.add
    local.set 22
    local.get 22
    local.set 3
    local.get 0
    local.get 3
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=68)
  (func (;4;) (type 0) (param i32 i32)
    (local i32)
    i32.const 0
    local.set 2
    loop  ;; label = @1
      local.get 0
      local.get 0
      local.get 0
      call 5
      local.get 2
      i32.const 1
      i32.add
      local.tee 2
      local.get 1
      i32.ne
      br_if 0 (;@1;)
    end)
  (func (;5;) (type 1) (param i32 i32 i32)
    local.get 0
    local.get 1
    local.get 2
    call 3
    local.get 0
    call 6)
  (func (;6;) (type 2) (param i32)
    (local i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64)
    local.get 0
    i32.load offset=32
    i64.extend_i32_u
    local.set 1
    local.get 1
    i64.const 22
    i64.shr_u
    local.get 0
    i32.load offset=36
    i64.extend_i32_u
    local.tee 1
    i64.const 7
    i64.shl
    i64.const 536870911
    i64.and
    i64.or
    local.set 2
    local.get 1
    i64.const 22
    i64.shr_u
    local.get 0
    i32.load offset=40
    i64.extend_i32_u
    local.tee 1
    i64.const 7
    i64.shl
    i64.const 536870911
    i64.and
    i64.or
    local.set 3
    local.get 1
    i64.const 22
    i64.shr_u
    local.get 0
    i32.load offset=44
    i64.extend_i32_u
    local.tee 1
    i64.const 7
    i64.shl
    i64.const 536870911
    i64.and
    i64.or
    local.set 4
    local.get 1
    i64.const 22
    i64.shr_u
    local.get 0
    i32.load offset=48
    i64.extend_i32_u
    local.tee 1
    i64.const 7
    i64.shl
    i64.const 536870911
    i64.and
    i64.or
    local.set 5
    local.get 1
    i64.const 22
    i64.shr_u
    local.get 0
    i32.load offset=52
    i64.extend_i32_u
    local.tee 1
    i64.const 7
    i64.shl
    i64.const 536870911
    i64.and
    i64.or
    local.set 6
    local.get 1
    i64.const 22
    i64.shr_u
    local.get 0
    i32.load offset=56
    i64.extend_i32_u
    local.tee 1
    i64.const 7
    i64.shl
    i64.const 536870911
    i64.and
    i64.or
    local.set 7
    local.get 1
    i64.const 22
    i64.shr_u
    local.get 0
    i32.load offset=60
    i64.extend_i32_u
    local.tee 1
    i64.const 7
    i64.shl
    i64.const 536870911
    i64.and
    i64.or
    local.set 8
    local.get 1
    i64.const 22
    i64.shr_u
    local.get 0
    i32.load offset=64
    i64.extend_i32_u
    local.tee 1
    i64.const 7
    i64.shl
    i64.const 536870911
    i64.and
    i64.or
    local.set 9
    local.get 1
    i64.const 22
    i64.shr_u
    local.get 0
    i32.load offset=68
    i64.extend_i32_u
    local.tee 1
    i64.const 7
    i64.shl
    i64.const 536870911
    i64.and
    i64.or
    local.set 10
    local.get 2
    i64.const 536870911
    i64.mul
    local.get 3
    i64.const 536870911
    i64.mul
    i64.add
    local.get 4
    i64.const 536870911
    i64.mul
    i64.add
    local.get 5
    i64.const 536800715
    i64.mul
    i64.add
    local.get 6
    i64.const 117700275
    i64.mul
    i64.add
    local.get 7
    i64.const 14453978
    i64.mul
    i64.add
    local.get 8
    i64.const 188500991
    i64.mul
    i64.add
    local.get 9
    i64.const 536870793
    i64.mul
    i64.add
    i64.const 29
    i64.shr_u
    local.get 2
    i64.const 536870911
    i64.mul
    i64.add
    local.get 3
    i64.const 536870911
    i64.mul
    i64.add
    local.get 4
    i64.const 536870911
    i64.mul
    i64.add
    local.get 5
    i64.const 536870911
    i64.mul
    i64.add
    local.get 6
    i64.const 536800715
    i64.mul
    i64.add
    local.get 7
    i64.const 117700275
    i64.mul
    i64.add
    local.get 8
    i64.const 14453978
    i64.mul
    i64.add
    local.get 9
    i64.const 188500991
    i64.mul
    i64.add
    local.get 10
    i64.const 536870793
    i64.mul
    i64.add
    i64.const 29
    i64.shr_u
    local.get 3
    i64.const 536870911
    i64.mul
    i64.add
    local.get 4
    i64.const 536870911
    i64.mul
    i64.add
    local.get 5
    i64.const 536870911
    i64.mul
    i64.add
    local.get 6
    i64.const 536870911
    i64.mul
    i64.add
    local.get 7
    i64.const 536800715
    i64.mul
    i64.add
    local.get 8
    i64.const 117700275
    i64.mul
    i64.add
    local.get 9
    i64.const 14453978
    i64.mul
    i64.add
    local.get 10
    i64.const 188500991
    i64.mul
    i64.add
    local.tee 1
    i64.const 536870911
    i64.and
    local.set 2
    local.get 1
    i64.const 29
    i64.shr_u
    local.get 4
    i64.const 536870911
    i64.mul
    i64.add
    local.get 5
    i64.const 536870911
    i64.mul
    i64.add
    local.get 6
    i64.const 536870911
    i64.mul
    i64.add
    local.get 7
    i64.const 536870911
    i64.mul
    i64.add
    local.get 8
    i64.const 536800715
    i64.mul
    i64.add
    local.get 9
    i64.const 117700275
    i64.mul
    i64.add
    local.get 10
    i64.const 14453978
    i64.mul
    i64.add
    local.tee 1
    i64.const 536870911
    i64.and
    local.set 3
    local.get 1
    i64.const 29
    i64.shr_u
    local.get 5
    i64.const 536870911
    i64.mul
    i64.add
    local.get 6
    i64.const 536870911
    i64.mul
    i64.add
    local.get 7
    i64.const 536870911
    i64.mul
    i64.add
    local.get 8
    i64.const 536870911
    i64.mul
    i64.add
    local.get 9
    i64.const 536800715
    i64.mul
    i64.add
    local.get 10
    i64.const 117700275
    i64.mul
    i64.add
    local.tee 1
    i64.const 536870911
    i64.and
    local.set 4
    local.get 1
    i64.const 29
    i64.shr_u
    local.get 6
    i64.const 536870911
    i64.mul
    i64.add
    local.get 7
    i64.const 536870911
    i64.mul
    i64.add
    local.get 8
    i64.const 536870911
    i64.mul
    i64.add
    local.get 9
    i64.const 536870911
    i64.mul
    i64.add
    local.get 10
    i64.const 536800715
    i64.mul
    i64.add
    local.tee 1
    i64.const 536870911
    i64.and
    local.set 5
    local.get 1
    i64.const 29
    i64.shr_u
    local.get 7
    i64.const 536870911
    i64.mul
    i64.add
    local.get 8
    i64.const 536870911
    i64.mul
    i64.add
    local.get 9
    i64.const 536870911
    i64.mul
    i64.add
    local.get 10
    i64.const 536870911
    i64.mul
    i64.add
    local.tee 1
    i64.const 536870911
    i64.and
    local.set 6
    local.get 1
    i64.const 29
    i64.shr_u
    local.get 8
    i64.const 536870911
    i64.mul
    i64.add
    local.get 9
    i64.const 536870911
    i64.mul
    i64.add
    local.get 10
    i64.const 536870911
    i64.mul
    i64.add
    local.tee 1
    i64.const 536870911
    i64.and
    local.set 7
    local.get 1
    i64.const 29
    i64.shr_u
    local.get 9
    i64.const 536870911
    i64.mul
    i64.add
    local.get 10
    i64.const 536870911
    i64.mul
    i64.add
    local.tee 1
    i64.const 536870911
    i64.and
    local.set 8
    local.get 1
    i64.const 29
    i64.shr_u
    local.get 10
    i64.const 536870911
    i64.mul
    i64.add
    local.tee 1
    i64.const 536870911
    i64.and
    local.set 9
    local.get 1
    i64.const 29
    i64.shr_u
    local.set 10
    local.get 2
    i64.const 1
    i64.mul
    local.set 11
    local.get 2
    i64.const 157910888
    i64.mul
    local.get 3
    i64.const 1
    i64.mul
    i64.add
    local.set 12
    local.get 2
    i64.const 322848486
    i64.mul
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.get 4
    i64.const 1
    i64.mul
    i64.add
    local.set 13
    local.get 2
    i64.const 221378578
    i64.mul
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.get 4
    i64.const 157910888
    i64.mul
    i64.add
    local.get 5
    i64.const 1
    i64.mul
    i64.add
    local.set 14
    local.get 2
    i64.const 548
    i64.mul
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.get 4
    i64.const 322848486
    i64.mul
    i64.add
    local.get 5
    i64.const 157910888
    i64.mul
    i64.add
    local.get 6
    i64.const 1
    i64.mul
    i64.add
    local.set 15
    local.get 2
    i64.const 0
    i64.mul
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.get 4
    i64.const 221378578
    i64.mul
    i64.add
    local.get 5
    i64.const 322848486
    i64.mul
    i64.add
    local.get 6
    i64.const 157910888
    i64.mul
    i64.add
    local.get 7
    i64.const 1
    i64.mul
    i64.add
    local.set 16
    local.get 2
    i64.const 0
    i64.mul
    local.get 3
    i64.const 0
    i64.mul
    i64.add
    local.get 4
    i64.const 548
    i64.mul
    i64.add
    local.get 5
    i64.const 221378578
    i64.mul
    i64.add
    local.get 6
    i64.const 322848486
    i64.mul
    i64.add
    local.get 7
    i64.const 157910888
    i64.mul
    i64.add
    local.get 8
    i64.const 1
    i64.mul
    i64.add
    local.set 17
    local.get 2
    i64.const 0
    i64.mul
    local.get 3
    i64.const 0
    i64.mul
    i64.add
    local.get 4
    i64.const 0
    i64.mul
    i64.add
    local.get 5
    i64.const 548
    i64.mul
    i64.add
    local.get 6
    i64.const 221378578
    i64.mul
    i64.add
    local.get 7
    i64.const 322848486
    i64.mul
    i64.add
    local.get 8
    i64.const 157910888
    i64.mul
    i64.add
    local.get 9
    i64.const 1
    i64.mul
    i64.add
    local.set 18
    local.get 2
    i64.const 4194304
    i64.mul
    local.get 3
    i64.const 0
    i64.mul
    i64.add
    local.get 4
    i64.const 0
    i64.mul
    i64.add
    local.get 5
    i64.const 0
    i64.mul
    i64.add
    local.get 6
    i64.const 548
    i64.mul
    i64.add
    local.get 7
    i64.const 221378578
    i64.mul
    i64.add
    local.get 8
    i64.const 322848486
    i64.mul
    i64.add
    local.get 9
    i64.const 157910888
    i64.mul
    i64.add
    local.get 10
    i64.const 1
    i64.mul
    i64.add
    local.set 19
    local.get 0
    i32.load
    i64.extend_i32_u
    local.get 11
    i64.sub
    local.set 1
    local.get 0
    local.get 1
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store
    local.get 1
    i64.const 29
    i64.shr_s
    local.get 0
    i32.load offset=4
    i64.extend_i32_u
    i64.add
    local.get 12
    i64.sub
    local.set 1
    local.get 0
    local.get 1
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=4
    local.get 1
    i64.const 29
    i64.shr_s
    local.get 0
    i32.load offset=8
    i64.extend_i32_u
    i64.add
    local.get 13
    i64.sub
    local.set 1
    local.get 0
    local.get 1
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=8
    local.get 1
    i64.const 29
    i64.shr_s
    local.get 0
    i32.load offset=12
    i64.extend_i32_u
    i64.add
    local.get 14
    i64.sub
    local.set 1
    local.get 0
    local.get 1
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=12
    local.get 1
    i64.const 29
    i64.shr_s
    local.get 0
    i32.load offset=16
    i64.extend_i32_u
    i64.add
    local.get 15
    i64.sub
    local.set 1
    local.get 0
    local.get 1
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=16
    local.get 1
    i64.const 29
    i64.shr_s
    local.get 0
    i32.load offset=20
    i64.extend_i32_u
    i64.add
    local.get 16
    i64.sub
    local.set 1
    local.get 0
    local.get 1
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=20
    local.get 1
    i64.const 29
    i64.shr_s
    local.get 0
    i32.load offset=24
    i64.extend_i32_u
    i64.add
    local.get 17
    i64.sub
    local.set 1
    local.get 0
    local.get 1
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=24
    local.get 1
    i64.const 29
    i64.shr_s
    local.get 0
    i32.load offset=28
    i64.extend_i32_u
    i64.add
    local.get 18
    i64.sub
    local.set 1
    local.get 0
    local.get 1
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=28
    local.get 1
    i64.const 29
    i64.shr_s
    local.get 0
    i32.load offset=32
    i64.extend_i32_u
    i64.add
    local.get 19
    i64.sub
    local.set 1
    local.get 0
    local.get 1
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=32
    local.get 0
    local.get 2
    i32.wrap_i64
    i32.store offset=36
    local.get 0
    local.get 3
    i32.wrap_i64
    i32.store offset=40
    local.get 0
    local.get 4
    i32.wrap_i64
    i32.store offset=44
    local.get 0
    local.get 5
    i32.wrap_i64
    i32.store offset=48
    local.get 0
    local.get 6
    i32.wrap_i64
    i32.store offset=52
    local.get 0
    local.get 7
    i32.wrap_i64
    i32.store offset=56
    local.get 0
    local.get 8
    i32.wrap_i64
    i32.store offset=60
    local.get 0
    local.get 9
    i32.wrap_i64
    i32.store offset=64
    local.get 0
    local.get 10
    i32.wrap_i64
    i32.store offset=68)
  (func (;7;) (type 0) (param i32 i32)
    (local i32)
    i32.const 0
    local.set 2
    loop  ;; label = @1
      local.get 0
      local.get 0
      call 8
      local.get 2
      i32.const 1
      i32.add
      local.tee 2
      local.get 1
      i32.ne
      br_if 0 (;@1;)
    end)
  (func (;8;) (type 0) (param i32 i32)
    (local i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64)
    local.get 1
    i32.load
    i64.extend_i32_u
    local.set 4
    local.get 1
    i32.load offset=4
    i64.extend_i32_u
    local.set 5
    local.get 1
    i32.load offset=8
    i64.extend_i32_u
    local.set 6
    local.get 1
    i32.load offset=12
    i64.extend_i32_u
    local.set 7
    local.get 1
    i32.load offset=16
    i64.extend_i32_u
    local.set 8
    local.get 1
    i32.load offset=20
    i64.extend_i32_u
    local.set 9
    local.get 1
    i32.load offset=24
    i64.extend_i32_u
    local.set 10
    local.get 1
    i32.load offset=28
    i64.extend_i32_u
    local.set 11
    local.get 1
    i32.load offset=32
    i64.extend_i32_u
    local.set 12
    local.get 4
    local.get 4
    i64.mul
    local.set 2
    i64.const 536870912
    local.get 2
    i64.const 536870911
    i64.and
    i64.sub
    local.set 3
    local.get 2
    local.get 3
    i64.add
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.set 13
    local.get 15
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.set 14
    local.get 16
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.set 15
    local.get 17
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.set 16
    local.get 18
    local.set 17
    local.get 19
    local.set 18
    local.get 20
    local.set 19
    local.get 3
    i64.const 4194304
    i64.mul
    local.set 20
    local.get 5
    local.get 4
    i64.mul
    i64.const 1
    i64.shl
    local.get 13
    i64.add
    local.set 2
    i64.const 536870912
    local.get 2
    i64.const 536870911
    i64.and
    i64.sub
    local.set 3
    local.get 2
    local.get 3
    i64.add
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.get 5
    local.get 5
    i64.mul
    i64.add
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.set 13
    local.get 15
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.set 14
    local.get 16
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.set 15
    local.get 17
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.set 16
    local.get 18
    local.set 17
    local.get 19
    local.set 18
    local.get 20
    local.set 19
    local.get 3
    i64.const 4194304
    i64.mul
    local.set 20
    local.get 6
    local.get 4
    i64.mul
    i64.const 1
    i64.shl
    local.get 13
    i64.add
    local.set 2
    i64.const 536870912
    local.get 2
    i64.const 536870911
    i64.and
    i64.sub
    local.set 3
    local.get 2
    local.get 3
    i64.add
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.get 6
    local.get 5
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.set 13
    local.get 15
    local.get 6
    local.get 6
    i64.mul
    i64.add
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.set 14
    local.get 16
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.set 15
    local.get 17
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.set 16
    local.get 18
    local.set 17
    local.get 19
    local.set 18
    local.get 20
    local.set 19
    local.get 3
    i64.const 4194304
    i64.mul
    local.set 20
    local.get 7
    local.get 4
    i64.mul
    i64.const 1
    i64.shl
    local.get 13
    i64.add
    local.set 2
    i64.const 536870912
    local.get 2
    i64.const 536870911
    i64.and
    i64.sub
    local.set 3
    local.get 2
    local.get 3
    i64.add
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.get 7
    local.get 5
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.set 13
    local.get 15
    local.get 7
    local.get 6
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.set 14
    local.get 16
    local.get 7
    local.get 7
    i64.mul
    i64.add
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.set 15
    local.get 17
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.set 16
    local.get 18
    local.set 17
    local.get 19
    local.set 18
    local.get 20
    local.set 19
    local.get 3
    i64.const 4194304
    i64.mul
    local.set 20
    local.get 8
    local.get 4
    i64.mul
    i64.const 1
    i64.shl
    local.get 13
    i64.add
    local.set 2
    i64.const 536870912
    local.get 2
    i64.const 536870911
    i64.and
    i64.sub
    local.set 3
    local.get 2
    local.get 3
    i64.add
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.get 8
    local.get 5
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.set 13
    local.get 15
    local.get 8
    local.get 6
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.set 14
    local.get 16
    local.get 8
    local.get 7
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.set 15
    local.get 17
    local.get 8
    local.get 8
    i64.mul
    i64.add
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.set 16
    local.get 18
    local.set 17
    local.get 19
    local.set 18
    local.get 20
    local.set 19
    local.get 3
    i64.const 4194304
    i64.mul
    local.set 20
    local.get 9
    local.get 4
    i64.mul
    i64.const 1
    i64.shl
    local.get 13
    i64.add
    local.set 2
    i64.const 536870912
    local.get 2
    i64.const 536870911
    i64.and
    i64.sub
    local.set 3
    local.get 2
    local.get 3
    i64.add
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.get 9
    local.get 5
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.set 13
    local.get 15
    local.get 9
    local.get 6
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.set 14
    local.get 16
    local.get 9
    local.get 7
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.set 15
    local.get 17
    local.get 9
    local.get 8
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.set 16
    local.get 18
    local.get 9
    local.get 9
    i64.mul
    i64.add
    local.set 17
    local.get 19
    local.set 18
    local.get 20
    local.set 19
    local.get 3
    i64.const 4194304
    i64.mul
    local.set 20
    local.get 10
    local.get 4
    i64.mul
    i64.const 1
    i64.shl
    local.get 13
    i64.add
    local.set 2
    i64.const 536870912
    local.get 2
    i64.const 536870911
    i64.and
    i64.sub
    local.set 3
    local.get 2
    local.get 3
    i64.add
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.get 10
    local.get 5
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.set 13
    local.get 15
    local.get 10
    local.get 6
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.set 14
    local.get 16
    local.get 10
    local.get 7
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.set 15
    local.get 17
    local.get 10
    local.get 8
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.set 16
    local.get 18
    local.get 10
    local.get 9
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.set 17
    local.get 19
    local.get 10
    local.get 10
    i64.mul
    i64.add
    local.set 18
    local.get 20
    local.set 19
    local.get 3
    i64.const 4194304
    i64.mul
    local.set 20
    local.get 11
    local.get 4
    i64.mul
    i64.const 1
    i64.shl
    local.get 13
    i64.add
    local.set 2
    i64.const 536870912
    local.get 2
    i64.const 536870911
    i64.and
    i64.sub
    local.set 3
    local.get 2
    local.get 3
    i64.add
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.get 11
    local.get 5
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.set 13
    local.get 15
    local.get 11
    local.get 6
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.set 14
    local.get 16
    local.get 11
    local.get 7
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.set 15
    local.get 17
    local.get 11
    local.get 8
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.set 16
    local.get 18
    local.get 11
    local.get 9
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.set 17
    local.get 19
    local.get 11
    local.get 10
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.set 18
    local.get 20
    local.get 11
    local.get 11
    i64.mul
    i64.add
    local.set 19
    local.get 3
    i64.const 4194304
    i64.mul
    local.set 20
    local.get 12
    local.get 4
    i64.mul
    i64.const 1
    i64.shl
    local.get 13
    i64.add
    local.set 2
    i64.const 536870912
    local.get 2
    i64.const 536870911
    i64.and
    i64.sub
    local.set 3
    local.get 2
    local.get 3
    i64.add
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.get 12
    local.get 5
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 157910888
    i64.mul
    i64.add
    local.set 13
    local.get 15
    local.get 12
    local.get 6
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 322848486
    i64.mul
    i64.add
    local.set 14
    local.get 16
    local.get 12
    local.get 7
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 221378578
    i64.mul
    i64.add
    local.set 15
    local.get 17
    local.get 12
    local.get 8
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.get 3
    i64.const 548
    i64.mul
    i64.add
    local.set 16
    local.get 18
    local.get 12
    local.get 9
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.set 17
    local.get 19
    local.get 12
    local.get 10
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.set 18
    local.get 20
    local.get 12
    local.get 11
    i64.mul
    i64.const 1
    i64.shl
    i64.add
    local.set 19
    local.get 12
    local.get 12
    i64.mul
    local.get 3
    i64.const 4194304
    i64.mul
    i64.add
    local.set 20
    local.get 0
    local.get 13
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store
    local.get 13
    i64.const 29
    i64.shr_u
    local.get 14
    i64.add
    local.set 14
    local.get 0
    local.get 14
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=4
    local.get 14
    i64.const 29
    i64.shr_u
    local.get 15
    i64.add
    local.set 15
    local.get 0
    local.get 15
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=8
    local.get 15
    i64.const 29
    i64.shr_u
    local.get 16
    i64.add
    local.set 16
    local.get 0
    local.get 16
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=12
    local.get 16
    i64.const 29
    i64.shr_u
    local.get 17
    i64.add
    local.set 17
    local.get 0
    local.get 17
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=16
    local.get 17
    i64.const 29
    i64.shr_u
    local.get 18
    i64.add
    local.set 18
    local.get 0
    local.get 18
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=20
    local.get 18
    i64.const 29
    i64.shr_u
    local.get 19
    i64.add
    local.set 19
    local.get 0
    local.get 19
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=24
    local.get 19
    i64.const 29
    i64.shr_u
    local.get 20
    i64.add
    local.set 20
    local.get 0
    local.get 20
    i64.const 536870911
    i64.and
    i32.wrap_i64
    i32.store offset=28
    local.get 20
    i64.const 29
    i64.shr_u
    local.get 21
    i64.add
    local.set 21
    local.get 0
    local.get 21
    i32.wrap_i64
    i32.store offset=32)
  (memory (;0;) 100)
  (export "benchMontgomery" (func 0))
  (export "benchSchoolbook" (func 2))
  (export "benchBarrett" (func 4))
  (export "benchSquare" (func 7))
  (export "memory" (memory 0)))
