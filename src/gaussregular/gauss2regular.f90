module gauss2regular_mod
  use iso_c_binding
  implicit none
contains

  logical function is_missing(v, missval)
    implicit none
    real(c_double), intent(in) :: v, missval

    if (missval /= missval) then
      is_missing = (v /= v)
    else
      is_missing = (v == missval)
    end if
  end function is_missing

  subroutine grib_get_reduced_row_c(pl, lon_first, lon_last, npoints, ilon_first, ilon_last) bind(C, name="grib_get_reduced_row_c")
    implicit none
    integer(c_int), value, intent(in) :: pl
    real(c_double), value, intent(in) :: lon_first, lon_last
    integer(c_int), intent(out) :: npoints, ilon_first, ilon_last

    real(c_double) :: range, dlon_first, dlon_last, lon_first_w
    integer(c_int) :: irange, ok

    range = lon_last - lon_first
    lon_first_w = lon_first
    if (range < 0.0d0) then
      range = range + 360.0d0
      lon_first_w = lon_first_w - 360.0d0
    end if

    npoints = int((range * dble(pl)) / 360.0d0) + 1
    ilon_first = int((lon_first_w * dble(pl)) / 360.0d0)
    ilon_last = int((lon_last * dble(pl)) / 360.0d0)

    irange = ilon_last - ilon_first + 1

    if (irange /= npoints) then
      if (irange > npoints) then
        dlon_first = dble(ilon_first) * 360.0d0 / dble(pl)
        if (dlon_first < lon_first_w) then
          ilon_first = ilon_first + 1
          irange = irange - 1
        end if

        dlon_last = dble(ilon_last) * 360.0d0 / dble(pl)
        if (dlon_last > lon_last) then
          ilon_last = ilon_last - 1
          irange = irange - 1
        end if
      else
        ok = 0

        dlon_first = dble(ilon_first - 1) * 360.0d0 / dble(pl)
        if (dlon_first > lon_first_w) then
          ilon_first = ilon_first - 1
          irange = irange + 1
          ok = 1
        end if

        dlon_last = dble(ilon_last + 1) * 360.0d0 / dble(pl)
        if (dlon_last < lon_last) then
          ilon_last = ilon_last + 1
          irange = irange + 1
          ok = 1
        end if

        if (ok == 0) npoints = npoints - 1
      end if
    else
      dlon_first = dble(ilon_first) * 360.0d0 / dble(pl)
      if (dlon_first < lon_first_w) then
        ilon_first = ilon_first + 1
        ilon_last = ilon_last + 1
      end if
    end if

    if (ilon_first < 0) ilon_first = ilon_first + pl
  end subroutine grib_get_reduced_row_c

  integer(c_int) function reduced_grid_is_global_c(npv, nxmax, xfirst, xlast) bind(C, name="reduced_grid_is_global_c")
    implicit none
    integer(c_int), value, intent(in) :: npv, nxmax
    real(c_double), value, intent(in) :: xfirst, xlast
    real(c_double) :: dx_global, dx_data

    if (npv > 0) then
      dx_global = 90.0d0 / dble(npv)
    else
      dx_global = 999.0d0
    end if

    dx_data = 360.0d0 - (xlast - xfirst)
    if ((dx_data > dx_global) .and. (dx_data * dble(nxmax) > 360.0d0)) dx_data = 360.0d0 / dble(nxmax)

    if (dx_data > dx_global) then
      reduced_grid_is_global_c = 0
    else
      reduced_grid_is_global_c = 1
    end if
  end function reduced_grid_is_global_c

  integer(c_int) function infer_nlon_c(pl, nlat, npv, xfirst, xlast) bind(C, name="infer_nlon_c")
    implicit none
    integer(c_int), intent(in) :: pl(*)
    integer(c_int), value, intent(in) :: nlat, npv
    real(c_double), value, intent(in) :: xfirst, xlast

    integer(c_int) :: j, nxmax
    integer(c_int) :: row_count, ilon_first, ilon_last
    integer(c_int) :: np4

    if (nlat <= 0) then
      infer_nlon_c = 0
      return
    end if

    if (npv > 0) then
      nxmax = 0
      do j = 1, nlat
        if (pl(j) > nxmax) nxmax = pl(j)
      end do

      if (reduced_grid_is_global_c(npv, nxmax, xfirst, xlast) /= 0) then
        infer_nlon_c = pl(nlat / 2 + 1)
      else
        np4 = npv * 4
        call grib_get_reduced_row_c(np4, xfirst, xlast, row_count, ilon_first, ilon_last)
        infer_nlon_c = row_count
      end if
    else
      infer_nlon_c = pl(nlat / 2 + 1)
    end if
  end function infer_nlon_c

  subroutine regularize_values_f64(values, nvals, pl, nlat, missval, nearest, nlon, out) bind(C, name="regularize_values_f64")
    implicit none
    integer(c_int), value, intent(in) :: nvals, nlat, nearest, nlon
    real(c_double), intent(in) :: values(*)
    integer(c_int), intent(in) :: pl(*)
    real(c_double), value, intent(in) :: missval
    real(c_double), intent(out) :: out(*)

    integer(c_int) :: j, i, k, src_n, ptr
    integer(c_int) :: i0, i1, idx, p, m, d
    real(c_double) :: scale, x, t, v0, v1, v
    logical :: has_missing, m0, m1, found, use_nearest

    if (nvals <= 0 .or. nlat <= 0 .or. nlon <= 0) return

    ptr = 1
    use_nearest = (nearest /= 0)

    do j = 1, nlat
      src_n = pl(j)
      has_missing = .false.
      do k = 0, src_n - 1
        if (is_missing(values(ptr + k), missval)) then
          has_missing = .true.
          exit
        end if
      end do

      if (src_n == nlon) then
        do i = 1, nlon
          out((j - 1) * nlon + i) = values(ptr + i - 1)
        end do
        ptr = ptr + src_n
        cycle
      end if

      scale = dble(src_n) / dble(nlon)

      do i = 1, nlon
        x = dble(i - 1) * scale

        if (use_nearest) then
          idx = int(floor(x + 0.5d0)) + 1
          if (idx > src_n) idx = idx - src_n
          v = values(ptr + idx - 1)

          if ((.not. has_missing) .or. (.not. is_missing(v, missval))) then
            out((j - 1) * nlon + i) = v
          else
            found = .false.
            do d = 1, src_n - 1
              p = idx + d
              m = idx - d
              if (p > src_n) p = p - src_n
              if (m < 1) m = m + src_n

              if (.not. is_missing(values(ptr + p - 1), missval)) then
                out((j - 1) * nlon + i) = values(ptr + p - 1)
                found = .true.
                exit
              end if

              if (.not. is_missing(values(ptr + m - 1), missval)) then
                out((j - 1) * nlon + i) = values(ptr + m - 1)
                found = .true.
                exit
              end if
            end do

            if (.not. found) out((j - 1) * nlon + i) = missval
          end if

        else
          i0 = int(floor(x)) + 1
          t = x - floor(x)
          i1 = i0 + 1
          if (i1 > src_n) i1 = i1 - src_n

          v0 = values(ptr + i0 - 1)
          v1 = values(ptr + i1 - 1)

          if (has_missing) then
            m0 = is_missing(v0, missval)
            m1 = is_missing(v1, missval)

            if (m0 .and. m1) then
              out((j - 1) * nlon + i) = missval
            else if (m0) then
              out((j - 1) * nlon + i) = v1
            else if (m1) then
              out((j - 1) * nlon + i) = v0
            else
              out((j - 1) * nlon + i) = v0 * (1.0d0 - t) + v1 * t
            end if
          else
            out((j - 1) * nlon + i) = v0 * (1.0d0 - t) + v1 * t
          end if
        end if
      end do

      ptr = ptr + src_n
    end do
  end subroutine regularize_values_f64

end module gauss2regular_mod
